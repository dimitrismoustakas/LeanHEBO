# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""End-to-end LeanHEBO optimizer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any, Self, cast

import torch

from leanhebo.acquisition import MACEEvaluator, PosteriorEvaluator, PosteriorStats
from leanhebo.checkpoint import load_checkpoint, save_checkpoint
from leanhebo.config import LeanHEBOConfig
from leanhebo.data import CandidateBatch, EncodedBatch
from leanhebo.data.store import ObservationStore
from leanhebo.diagnostics import Diagnostics, FitReport
from leanhebo.errors import CheckpointError, SearchSpaceExhaustedError
from leanhebo.gp import ConditionalExactGPSurrogate, ExactGPSurrogate
from leanhebo.runtime.rng import RandomStreams
from leanhebo.search import (
    ConditionalTorchNSGA2,
    MixedVariableSpec,
    NSGA2Result,
    TorchNSGA2,
    eliminate_semantic_duplicates,
)
from leanhebo.space import (
    Bool,
    Categorical,
    CompiledSpace,
    FixedInput,
    Float,
    Integer,
    Parameter,
    Space,
)
from leanhebo.transforms import OutputTransform


class _CompiledConditionalSearchSemantics:
    """Bind compiled conditional semantics and a fixed context to one search call."""

    def __init__(
        self,
        space: CompiledSpace,
        fixed: FixedInput | None,
        *,
        device: torch.device,
    ) -> None:
        semantics = space.conditional_semantics
        if semantics is None:
            raise ValueError("conditional search semantics require a conditional space")
        self.space = space
        self.fixed = fixed
        self._semantics = semantics
        dense_parameter_indices = (
            semantics.continuous_parameter_indices + semantics.categorical_parameter_indices
        )
        self._dense_parameter_indices = torch.tensor(
            dense_parameter_indices,
            dtype=torch.int64,
            device=device,
        )

    def _encoded(self, population: torch.Tensor) -> EncodedBatch:
        values = population.to(dtype=self.space.dtype)
        return EncodedBatch(
            values[:, : self.space.n_continuous],
            values[:, self.space.n_continuous :].to(torch.int64),
        )

    def activity_mask(self, population: torch.Tensor) -> torch.Tensor:
        activity = self._semantics.activity(self._encoded(population)).parameter
        return activity.index_select(1, self._dense_parameter_indices)

    def semantic_keys(self, population: torch.Tensor) -> torch.Tensor:
        return self._semantics.key_tensor(self._encoded(population))

    def finite_completion(self, count: int, *, existing: torch.Tensor) -> torch.Tensor:
        if count <= 0 or not self.space.context_is_finite(self.fixed):
            return existing.new_empty((0, self.space.dense_dimension))

        completed = existing.new_empty((0, self.space.dense_dimension))
        records = self.space.iter_contextual_records(self.fixed)
        chunk_size = max(64, min(1024, count * 4))
        while completed.shape[0] < count:
            chunk = list(islice(records, chunk_size))
            if not chunk:
                break
            encoded = self.space.encode(chunk).to(existing.device, dtype=existing.dtype)
            if self.fixed is not None:
                encoded = self.space.apply_fixed(encoded, self.fixed)
            candidates = eliminate_semantic_duplicates(
                encoded.to_dense(),
                self,
                existing=torch.cat((existing, completed), dim=0),
            )
            completed = torch.cat(
                (completed, candidates[: count - completed.shape[0]]),
                dim=0,
            )
        return completed


class LeanHEBO:
    """Single-objective HEBO with persistent tensor-native runtime state."""

    def __init__(
        self,
        space: Space | CompiledSpace,
        *,
        config: LeanHEBOConfig | None = None,
    ) -> None:
        self.config = LeanHEBOConfig() if config is None else config
        self.device = torch.device(self.config.runtime.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device {self.device} was requested but CUDA is unavailable")
        self.dtype: torch.dtype = getattr(torch, self.config.runtime.dtype)
        if isinstance(space, Space):
            self.public_space = space
            self.space = space.compile(dtype=self.dtype)
        else:
            self.public_space = Space(*space.parameters)
            self.space = (
                space
                if space.dtype == self.dtype
                else CompiledSpace(space.parameters, dtype=self.dtype)
            )
        self.diagnostics = Diagnostics(self.config.runtime)
        self.store = ObservationStore(
            self.space,
            device=self.device,
            nonfinite=self.config.nonfinite_policy,
        )
        self.output_transform = OutputTransform(cast(Any, self.config.warp)).to(
            device=self.device, dtype=self.dtype
        )
        self.random = RandomStreams.create(
            self.device,
            self.config.runtime.seed,
            self.config.search.seed,
        )
        self._sobol = self._new_sobol_engine()
        self._sobol_draw_count = 0
        self._surrogate: ExactGPSurrogate | None = None
        self._model_observation_version = -1
        self._force_full_refit = False
        self._previous_population: torch.Tensor | None = None
        self._last_search: NSGA2Result | None = None

    @property
    def random_samples(self) -> int:
        if self.space.dense_dimension == 0:
            return 1
        configured = self.config.random_samples
        return max(2, 1 + self.space.dense_dimension) if configured is None else configured

    @property
    def observations(self) -> int:
        return len(self.store)

    @property
    def surrogate(self) -> ExactGPSurrogate | None:
        return self._surrogate

    @property
    def last_search(self) -> NSGA2Result | None:
        return self._last_search

    def _new_sobol_engine(self) -> torch.quasirandom.SobolEngine | None:
        if self.space.dense_dimension == 0:
            return None
        return torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
            self.space.dense_dimension,
            scramble=True,
            seed=self.random.sobol_seed,
        )

    def _draw_sobol(
        self,
        count: int,
        fixed: FixedInput | None,
    ) -> CandidateBatch:
        if count < 0:
            raise ValueError("suggestion count cannot be negative")
        if self._sobol is None:
            unit = torch.empty((count, 0), dtype=self.dtype)
        else:
            unit = self._sobol.draw(count, dtype=self.dtype)
            self._sobol_draw_count += count
        dense = self.space.dense_from_unit(unit).to(device=self.device, dtype=self.dtype)
        return self.space.candidate_from_dense(dense, fixed=fixed)

    def suggest(
        self,
        n_suggestions: int = 1,
        fix_input: Mapping[str, object] | FixedInput | None = None,
    ) -> CandidateBatch:
        """Suggest a unique candidate batch, applying optional contextual assignments."""

        if isinstance(n_suggestions, bool) or not isinstance(n_suggestions, int):
            raise TypeError("n_suggestions must be an integer")
        if n_suggestions < 1:
            raise ValueError("n_suggestions must be positive")
        fixed = self._compile_fixed(fix_input)
        with self.diagnostics.phase("suggest.total"):
            if self.space.dense_dimension == 0:
                with self.diagnostics.phase("suggest.initial_sampling"):
                    return self._fill_unique(None, n_suggestions, fixed)
            if len(self.store) < self.random_samples:
                with self.diagnostics.phase("suggest.initial_sampling"):
                    return self._fill_unique(None, n_suggestions, fixed)
            self._ensure_model()
            return self._model_suggest(n_suggestions, fixed)

    def _compile_fixed(self, value: Mapping[str, object] | FixedInput | None) -> FixedInput | None:
        if value is None:
            return None
        fixed = self.space.compile_fixed(value) if isinstance(value, Mapping) else value
        if fixed.space_fingerprint != self.space.fingerprint:
            raise ValueError("FixedInput belongs to a different design space")
        return fixed

    def _make_surrogate(self) -> ExactGPSurrogate:
        if self.space.is_conditional:
            return ConditionalExactGPSurrogate(
                space=self.space,
                config=self.config.gp,
                runtime=self.config.runtime,
                generator=self.random.model,
                diagnostics=self.diagnostics,
            )
        category_sizes = tuple(
            round(parameter.optimization_bounds[1] - parameter.optimization_bounds[0]) + 1
            for parameter in self.space.categorical_parameters
        )
        return ExactGPSurrogate(
            num_continuous=self.space.n_continuous,
            category_sizes=category_sizes,
            config=self.config.gp,
            runtime=self.config.runtime,
            generator=self.random.model,
            diagnostics=self.diagnostics,
        )

    def _ensure_model(self) -> FitReport | None:
        if (
            self._surrogate is not None
            and self._model_observation_version == self.store.observation_version
            and not self._force_full_refit
        ):
            return None
        observations = self.store._materialize_view()
        if len(observations) < 2:
            raise RuntimeError("at least two finite observations are required for model fitting")
        with self.diagnostics.phase("suggest.fit_output_transform"):
            transformed = self.output_transform.fit_transform(observations.y)
        if self._surrogate is None:
            with self.diagnostics.phase("suggest.gp_construct"):
                self._surrogate = self._make_surrogate()
        phase = (
            "suggest.gp_full_refit"
            if self._force_full_refit
            else ("suggest.gp_initial_fit" if not self._surrogate.fitted else "suggest.gp_update")
        )
        with self.diagnostics.phase(phase):
            report = self._surrogate.fit(
                observations.continuous,
                observations.categorical,
                transformed,
                transform_version=self.output_transform.version,
                force_full_refit=self._force_full_refit,
            )
        self._model_observation_version = self.store.observation_version
        self._force_full_refit = False
        return report

    def _model_suggest(self, n_suggestions: int, fixed: FixedInput | None) -> CandidateBatch:
        assert self._surrogate is not None
        posterior = PosteriorEvaluator(
            self._surrogate,
            batch_size=self.config.runtime.acquisition_batch_size,
        )
        incumbent = self._incumbent(fixed)
        incumbent_stats = posterior.evaluate(incumbent.continuous, incumbent.categorical)
        kappa = self._kappa(n_suggestions)
        mace = MACEEvaluator(
            posterior,
            best_y=incumbent_stats.mean[0],
            kappa=kappa,
            epsilon=self.config.acquisition.epsilon,
            stochastic=self.config.acquisition.stochastic,
            generator=self.random.acquisition,
        )

        def objective(dense: torch.Tensor) -> torch.Tensor:
            # NSGA-II already repairs its populations, but encoded conversion must still handle
            # non-uniform domains such as logarithmic integers. Fuse that final canonicalization
            # with the dense split instead of materializing and validating a second dense tensor.
            encoded = self.space.encoded_from_dense(dense, repair=True, fixed=fixed)
            return mace(encoded.continuous, encoded.categorical)

        search_spec = self._search_spec(fixed)
        search_options: dict[str, Any] = {
            "population_size": self.config.search.population_size,
            "generations": self.config.search.generations,
            "crossover_probability": self.config.search.crossover_probability,
            "crossover_eta": self.config.search.crossover_eta,
            "mutation_probability": self.config.search.mutation_probability,
            "mutation_eta": self.config.search.mutation_eta,
            "tournament_size": self.config.search.tournament_size,
            "eliminate_duplicate_points": self.config.search.eliminate_duplicates,
        }
        search: TorchNSGA2
        if self.space.is_conditional:
            search = ConditionalTorchNSGA2(
                search_spec,
                _CompiledConditionalSearchSemantics(
                    self.space,
                    fixed,
                    device=self.device,
                ),
                **search_options,
            )
        else:
            search = TorchNSGA2(search_spec, **search_options)
        previous = (
            self._previous_population if self.config.search.reuse_previous_population else None
        )
        with self.diagnostics.phase("suggest.search.total"):
            result = search.minimize(
                objective,
                incumbents=self.space.to_dense(incumbent),
                initial_population=previous,
                generator=self.random.search,
            )
        self._last_search = result
        self._previous_population = (
            result.population.detach().clone()
            if self.config.search.reuse_previous_population
            else None
        )
        # A non-empty population always has a rank-zero front.
        pool = result.pareto_population
        candidates = self.space.candidate_from_dense(pool, fixed=fixed)
        unique = self.store.unique_mask(candidates)
        candidates = candidates.select(unique)
        if len(candidates):
            with self.diagnostics.phase("suggest.selection"):
                stats = posterior.evaluate(candidates.continuous, candidates.categorical)
                selected_indices = self._selection_indices(
                    stats, min(n_suggestions, len(candidates))
                )
                selected = candidates.select(selected_indices)
        else:
            selected = None
        with self.diagnostics.phase("suggest.uniqueness"):
            return self._fill_unique(selected, n_suggestions, fixed)

    def _incumbent(self, fixed: FixedInput | None) -> CandidateBatch:
        observations = self.store._materialize_view()
        eligible = torch.ones(len(observations), dtype=torch.bool, device=self.device)
        if fixed is not None:
            if self.space.is_conditional:
                semantics = self.space.conditional_semantics
                assert semantics is not None
                parameter_activity = semantics.activity(observations.encoded).parameter
                if fixed.continuous_indices.numel():
                    indices = fixed.continuous_indices.to(self.device)
                    values = fixed.continuous_values.to(self.device, dtype=self.dtype)
                    parameter_indices = torch.tensor(
                        semantics.continuous_parameter_indices,
                        dtype=torch.int64,
                        device=self.device,
                    ).index_select(0, indices)
                    active = parameter_activity.index_select(1, parameter_indices)
                    matches = observations.continuous[:, indices] == values
                    eligible &= ((~active) | matches).all(dim=1)
                if fixed.categorical_indices.numel():
                    indices = fixed.categorical_indices.to(self.device)
                    values = fixed.categorical_values.to(self.device)
                    parameter_indices = torch.tensor(
                        semantics.categorical_parameter_indices,
                        dtype=torch.int64,
                        device=self.device,
                    ).index_select(0, indices)
                    active = parameter_activity.index_select(1, parameter_indices)
                    matches = observations.categorical[:, indices] == values
                    eligible &= ((~active) | matches).all(dim=1)
            else:
                if fixed.continuous_indices.numel():
                    indices = fixed.continuous_indices.to(self.device)
                    values = fixed.continuous_values.to(self.device, dtype=self.dtype)
                    eligible &= (observations.continuous[:, indices] == values).all(dim=1)
                if fixed.categorical_indices.numel():
                    indices = fixed.categorical_indices.to(self.device)
                    values = fixed.categorical_values.to(self.device)
                    eligible &= (observations.categorical[:, indices] == values).all(dim=1)
            if not bool(eligible.any()):
                eligible.fill_(True)
        scores = torch.where(eligible, observations.y, torch.inf)
        index = scores.argmin().reshape(1)
        encoded = observations.encoded.select(index)
        if fixed is not None:
            encoded = self.space.apply_fixed(encoded, fixed)
        return self.space.decode(encoded, fixed=fixed)

    def _kappa(self, n_suggestions: int) -> float:
        configured = self.config.acquisition.kappa
        if configured is not None:
            return configured
        iteration = max(1, len(self.store) // n_suggestions)
        dimension = self.space.dense_dimension
        inside = (2.0 + dimension / 2.0) * math.log(iteration) + math.log(
            3.0 * math.pi**2 / (3.0 * self.config.acquisition.delta)
        )
        return math.sqrt(self.config.acquisition.upsi * 2.0 * inside)

    def _search_spec(self, fixed: FixedInput | None) -> MixedVariableSpec:
        lower = self.space.dense_lower_bounds.to(self.device)
        upper = self.space.dense_upper_bounds.to(self.device)
        integer = (self.space.rounding_mask & ~self.space.categorical_mask).to(self.device)
        categorical = self.space.categorical_mask.to(self.device)
        steps = integer.to(dtype=self.dtype)
        fixed_mask = torch.zeros_like(integer)
        fixed_values = lower.clone()
        if fixed is not None:
            fixed_mask = fixed.dense_mask.to(self.device)
            if fixed.continuous_indices.numel():
                indices = fixed.continuous_indices.to(self.device)
                fixed_values[indices] = fixed.continuous_values.to(self.device, dtype=self.dtype)
            if fixed.categorical_indices.numel():
                indices = fixed.categorical_indices.to(self.device)
                fixed_values[self.space.n_continuous + indices] = fixed.categorical_values.to(
                    self.device, dtype=self.dtype
                )
        return MixedVariableSpec(
            lower,
            upper,
            integer_mask=integer,
            categorical_mask=categorical,
            steps=steps,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )

    def _selection_indices(self, stats: PosteriorStats, count: int) -> torch.Tensor:
        population = stats.mean.numel()
        if count < 1 or count > population:
            raise ValueError("invalid candidate selection count")
        random_order = torch.randperm(
            population,
            device=stats.mean.device,
            generator=self.random.selection,
        ).tolist()
        selected: list[int] = []
        if count > 2:
            selected.append(int(stats.stddev.argmax().item()))
            best_mean = int(stats.mean.argmin().item())
            if best_mean not in selected:
                selected.append(best_mean)
        selected.extend(index for index in random_order if index not in selected)
        return torch.tensor(selected[:count], device=stats.mean.device, dtype=torch.int64)

    def _fill_unique(
        self,
        existing: CandidateBatch | None,
        requested: int,
        fixed: FixedInput | None,
    ) -> CandidateBatch:
        encoded_rows: list[EncodedBatch] = []
        seen: set[tuple[int, ...]] = set()
        if existing is not None:
            take = min(len(existing), requested)
            initial = existing.select(slice(0, take))
            encoded_rows.append(initial.encoded)
            seen.update(self.space.canonical_keys(initial))
        retained = sum(len(batch) for batch in encoded_rows)
        fixed_values = {} if fixed is None else dict(fixed.decoded_values)
        finite_domain = (
            self.space.context_is_finite(fixed)
            if self.space.is_conditional
            else all(
                self._finite_cardinality(parameter, fixed_values) is not None
                for parameter in self.space.parameters
            )
        )
        while retained < requested:
            missing = requested - retained
            if self.space.dense_dimension == 0:
                completion = self._finite_unseen(missing, fixed, seen)
                encoded_rows.extend(completion)
                retained += sum(len(batch) for batch in completion)
                break

            draw = self._draw_sobol(missing, fixed)
            historical_unique = self.store.unique_mask(draw)
            keys = self.space.canonical_keys(draw)
            fallback: list[tuple[EncodedBatch, tuple[int, ...]]] = []
            fallback_index = 0
            exhausted = False
            for index, key in enumerate(keys):
                if bool(historical_unique[index]) and key not in seen:
                    row = draw.select([index]).encoded
                elif finite_domain:
                    if not fallback:
                        completion = self._finite_unseen(len(keys) - index, fixed, set(seen))
                        fallback = [
                            (batch.select([fallback_row]), fallback_key)
                            for batch in completion
                            for fallback_row, fallback_key in enumerate(
                                self.space.canonical_keys(batch)
                            )
                        ]
                    while fallback_index < len(fallback) and fallback[fallback_index][1] in seen:
                        fallback_index += 1
                    if fallback_index == len(fallback):
                        exhausted = True
                        break
                    row, key = fallback[fallback_index]
                    fallback_index += 1
                else:
                    continue
                encoded_rows.append(row)
                seen.add(key)
                retained += 1
            if exhausted:
                break
        if retained < requested:
            missing = requested - retained
            self.diagnostics.increment("suggest.uniqueness_exhausted", missing)
            raise SearchSpaceExhaustedError(
                f"could not produce {requested} unique candidate(s): "
                f"only {retained} unseen point(s) were found"
            )
        continuous = torch.cat([batch.continuous for batch in encoded_rows], dim=0)
        categorical = torch.cat([batch.categorical for batch in encoded_rows], dim=0)
        return self.space.decode(
            EncodedBatch(continuous, categorical),
            fixed=fixed,
        ).select(slice(0, requested))

    @staticmethod
    def _finite_cardinality(
        parameter: Parameter,
        fixed_values: Mapping[str, object],
    ) -> int | None:
        if parameter.name in fixed_values:
            return 1
        if isinstance(parameter, Float):
            return None
        if isinstance(parameter, Integer):
            if parameter.exponent:
                lower, upper = parameter.optimization_bounds
                return round(upper - lower) + 1
            return (parameter.high - parameter.low) // parameter.step + 1
        if isinstance(parameter, Categorical):
            return len(parameter.categories)
        if isinstance(parameter, Bool):
            return 2
        raise TypeError(f"unsupported parameter type {type(parameter).__name__}")

    @staticmethod
    def _finite_value(
        parameter: Parameter,
        index: int,
        fixed_values: Mapping[str, object],
    ) -> object:
        if parameter.name in fixed_values:
            return fixed_values[parameter.name]
        if isinstance(parameter, Integer):
            if parameter.exponent:
                lower, _ = parameter.optimization_bounds
                return int(parameter.base) ** (round(lower) + index)
            return parameter.low + index * parameter.step
        if isinstance(parameter, Categorical):
            return parameter.categories[index]
        if isinstance(parameter, Bool):
            return bool(index)
        raise TypeError(f"parameter {parameter.name!r} does not have a finite domain")

    def _finite_unseen(
        self,
        requested: int,
        fixed: FixedInput | None,
        seen: set[tuple[int, ...]],
    ) -> list[EncodedBatch]:
        """Return deterministic unseen rows when the contextual domain is finite."""

        if requested == 0:
            return []
        if self.space.is_conditional:
            return self._conditional_finite_unseen(requested, fixed, seen)
        fixed_values = {} if fixed is None else dict(fixed.decoded_values)
        cardinalities: list[int] = []
        for parameter in self.space.parameters:
            cardinality = self._finite_cardinality(parameter, fixed_values)
            if cardinality is None:
                return []
            cardinalities.append(cardinality)

        blocked = set(self.store.key_snapshot()) | seen
        completed: list[EncodedBatch] = []
        completed_count = 0
        total = math.prod(cardinalities)
        chunk_size = max(64, min(1024, requested * 4))
        for start in range(0, total, chunk_size):
            records: list[dict[str, object]] = []
            for ordinal in range(start, min(start + chunk_size, total)):
                remainder = ordinal
                indices = [0] * len(cardinalities)
                for position in range(len(cardinalities) - 1, -1, -1):
                    remainder, indices[position] = divmod(remainder, cardinalities[position])
                records.append(
                    {
                        parameter.name: self._finite_value(
                            parameter,
                            index,
                            fixed_values,
                        )
                        for parameter, index in zip(
                            self.space.parameters,
                            indices,
                            strict=True,
                        )
                    }
                )
            encoded = self.space.encode(records).to(self.device, dtype=self.dtype)
            keep: list[int] = []
            for index, key in enumerate(self.space.canonical_keys(encoded)):
                if key in blocked:
                    continue
                keep.append(index)
                blocked.add(key)
                seen.add(key)
                if len(keep) + completed_count == requested:
                    break
            if keep:
                completed.append(encoded.select(keep))
                completed_count += len(keep)
            if completed_count == requested:
                break
        return completed

    def _conditional_finite_unseen(
        self,
        requested: int,
        fixed: FixedInput | None,
        seen: set[tuple[int, ...]],
    ) -> list[EncodedBatch]:
        blocked = set(self.store.key_snapshot()) | seen
        completed: list[EncodedBatch] = []
        completed_count = 0
        records = self.space.iter_contextual_records(fixed)
        chunk_size = max(64, min(1024, requested * 4))
        while completed_count < requested:
            chunk = list(islice(records, chunk_size))
            if not chunk:
                break
            encoded = self.space.encode(chunk).to(self.device, dtype=self.dtype)
            if fixed is not None:
                encoded = self.space.apply_fixed(encoded, fixed)
            keep: list[int] = []
            for index, key in enumerate(self.space.canonical_keys(encoded)):
                if key in blocked:
                    continue
                keep.append(index)
                blocked.add(key)
                seen.add(key)
                if len(keep) + completed_count == requested:
                    break
            if keep:
                completed.append(encoded.select(keep))
                completed_count += len(keep)
        return completed

    def observe(self, candidates: CandidateBatch | EncodedBatch | object, y: object) -> int:
        """Append finite objective observations without doing eager GP work."""

        with self.diagnostics.phase("observe.total"):
            discarded_before = self.store.discarded_count
            retained = self.store.append(candidates, y)  # type: ignore[arg-type]
            discarded = self.store.discarded_count - discarded_before
            self.diagnostics.increment("observe.received", retained + discarded)
            self.diagnostics.increment("observe.discarded", discarded)
            return retained

    def refit(self) -> FitReport | None:
        """Run an explicit full GP refit now when model-based data are available."""

        if self.space.dense_dimension == 0 or len(self.store) < self.random_samples:
            return None
        self._force_full_refit = True
        return self._ensure_model()

    @property
    def best_x(self) -> CandidateBatch:
        if len(self.store) == 0:
            raise RuntimeError("no data has been observed")
        observations = self.store._materialize_view()
        index = observations.y.argmin().reshape(1)
        return self.space.decode(observations.encoded.select(index))

    @property
    def best_y(self) -> float:
        if len(self.store) == 0:
            raise RuntimeError("no data has been observed")
        return float(self.store._materialize_view().y.min().detach().cpu())

    def _checkpoint_state(self) -> dict[str, Any]:
        observations = self.store._materialize_view()
        return {
            "config": self.config.to_dict(),
            "space": self.public_space.to_spec(),
            "observations": {
                "continuous": observations.continuous.detach().cpu(),
                "categorical": observations.categorical.detach().cpu(),
                "y": observations.y.detach().cpu(),
            },
            "discarded_count": self.store.discarded_count,
            "output_transform": self.output_transform.state_dict(),
            "surrogate": (None if self._surrogate is None else self._surrogate.state_dict()),
            "model_current": (
                self._surrogate is not None
                and self._model_observation_version == self.store.observation_version
            ),
            "random": self.random.state_dict(),
            "sobol_draw_count": self._sobol_draw_count,
            "previous_population": (
                None
                if self._previous_population is None
                else self._previous_population.detach().cpu().clone()
            ),
        }

    def _restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.store.clear()
        observations = state["observations"]
        encoded = EncodedBatch(observations["continuous"], observations["categorical"])
        if len(encoded):
            self.store.append(encoded, observations["y"])
        discarded = state["discarded_count"]
        if isinstance(discarded, bool) or not isinstance(discarded, int) or discarded < 0:
            raise ValueError("checkpoint discarded-observation count is invalid")
        self.store.discarded_count = discarded
        self.output_transform.load_state_dict(state["output_transform"])
        self.output_transform.to(device=self.device, dtype=self.dtype)
        surrogate_state = state["surrogate"]
        if surrogate_state is not None:
            self._surrogate = self._make_surrogate()
            self._surrogate.load_state_dict(surrogate_state)
        else:
            self._surrogate = None
        self.random.load_state_dict(dict(state["random"]))
        self._sobol = self._new_sobol_engine()
        self._sobol_draw_count = int(state["sobol_draw_count"])
        if self._sobol_draw_count:
            assert self._sobol is not None
            self._sobol.fast_forward(  # type: ignore[no-untyped-call]
                self._sobol_draw_count
            )
        self._model_observation_version = (
            self.store.observation_version if state["model_current"] else -1
        )
        previous = state["previous_population"]
        self._previous_population = (
            None
            if previous is None or not self.config.search.reuse_previous_population
            else torch.as_tensor(previous, device=self.device, dtype=self.dtype).clone()
        )
        self._force_full_refit = False
        self._last_search = None

    def save(self, path: str | Path) -> None:
        save_checkpoint(path, self._checkpoint_state())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> Self:
        state = load_checkpoint(path, map_location=map_location)
        try:
            config = LeanHEBOConfig.from_dict(state["config"])
            if map_location is not None:
                config = replace(
                    config,
                    runtime=replace(config.runtime, device=str(torch.device(map_location))),
                )
            space = Space.from_spec(state["space"])
            optimizer = cls(space, config=config)
            optimizer._restore_checkpoint_state(state)
            return optimizer
        except CheckpointError:
            raise
        except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as error:
            raise CheckpointError(
                f"checkpoint payload is malformed or incompatible: {error}"
            ) from error
