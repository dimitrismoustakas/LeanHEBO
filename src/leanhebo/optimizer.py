# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""End-to-end LeanHEBO optimizer."""

from __future__ import annotations

import math
import platform
from collections.abc import Mapping
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any, Self, cast

import torch

from leanhebo.acquisition import MACEEvaluator, PosteriorEvaluator, PosteriorStats
from leanhebo.checkpoint import load_checkpoint, save_checkpoint
from leanhebo.config import LeanHEBOConfig
from leanhebo.data import CandidateBatch, EncodedBatch, ObservationStore
from leanhebo.diagnostics import Diagnostics, FitReport
from leanhebo.errors import CheckpointError, NumericalError, SearchSpaceExhaustedError
from leanhebo.gp import ExactGPSurrogate
from leanhebo.runtime.rng import RandomStreams
from leanhebo.search import MixedVariableSpec, NSGA2Result, TorchNSGA2
from leanhebo.space import CompiledSpace, FixedInput, Space
from leanhebo.transforms import OutputTransform

_STATE_SCHEMA_VERSION = 2


class LeanHEBO:
    """Single-objective HEBO with persistent tensor-native runtime state."""

    support_parallel_opt = True
    support_combinatorial = True
    support_contextual = True

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
        if self.config.runtime.deterministic:
            # This is deliberately opt-in because Torch's deterministic policy is process-global.
            torch.use_deterministic_algorithms(True)
        self.diagnostics = Diagnostics(self.config.runtime)
        self.store = ObservationStore(
            self.space,
            device=self.device,
            nonfinite=self.config.nonfinite_policy,
            retain_decoded=False,
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
        self._search_history: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._last_search: NSGA2Result | None = None

    @property
    def random_samples(self) -> int:
        configured = self.config.random_samples
        return max(2, 1 + len(self.space)) if configured is None else configured

    @property
    def observations(self) -> int:
        return len(self.store)

    @property
    def surrogate(self) -> ExactGPSurrogate | None:
        return self._surrogate

    @property
    def last_search(self) -> NSGA2Result | None:
        return self._last_search

    @property
    def search_history(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Return defensive snapshots retained when ``SearchConfig.keep_history`` is enabled."""

        return tuple(
            (population.clone(), objectives.clone())
            for population, objectives in self._search_history
        )

    def _new_sobol_engine(self) -> torch.quasirandom.SobolEngine:
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
        unit = self._sobol.draw(count, dtype=self.dtype)
        self._sobol_draw_count += count
        lower = self.space.dense_lower_bounds
        upper = self.space.dense_upper_bounds
        dense = lower + unit * (upper - lower)
        discrete = self.space.rounding_mask
        if bool(discrete.any()):
            cardinality = (upper[discrete] - lower[discrete]).round() + 1
            dense[:, discrete] = (
                torch.floor(unit[:, discrete] * cardinality).clamp_max(cardinality - 1)
                + lower[discrete]
            )
        dense = dense.to(device=self.device, dtype=self.dtype)
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
            if len(self.store) < self.random_samples:
                with self.diagnostics.phase("suggest.initial_sampling"):
                    return self._fill_unique(None, n_suggestions, fixed)
            try:
                self._ensure_model()
                return self._model_suggest(n_suggestions, fixed)
            except NumericalError:
                self.diagnostics.increment("gp.numerical_recovery_attempts")
                try:
                    self._ensure_model(reset_parameters=True)
                    return self._model_suggest(n_suggestions, fixed)
                except NumericalError:
                    self.diagnostics.increment("gp.numerical_recovery_failures")
                    raise

    def _compile_fixed(self, value: Mapping[str, object] | FixedInput | None) -> FixedInput | None:
        if value is None:
            return None
        fixed = self.space.compile_fixed(value) if isinstance(value, Mapping) else value
        if fixed.space_fingerprint != self.space.fingerprint:
            raise ValueError("FixedInput belongs to a different design space")
        return fixed

    def _make_surrogate(self) -> ExactGPSurrogate:
        category_sizes = tuple(
            round(parameter.optimization_bounds[1] - parameter.optimization_bounds[0]) + 1
            for parameter in self.space.parameters
            if parameter.is_categorical
        )
        return ExactGPSurrogate(
            num_continuous=self.space.n_continuous,
            category_sizes=category_sizes,
            config=self.config.gp,
            runtime=self.config.runtime,
            generator=self.random.model,
            diagnostics=self.diagnostics,
        )

    def _ensure_model(self, *, reset_parameters: bool = False) -> FitReport | None:
        if (
            self._surrogate is not None
            and self._model_observation_version == self.store.observation_version
            and not self._force_full_refit
            and not reset_parameters
        ):
            return None
        observations = self.store._materialize_view()
        if len(observations) < 2:
            raise RuntimeError("at least two finite observations are required for model fitting")
        with self.diagnostics.phase("suggest.fit_output_transform"):
            transformed = self.output_transform.fit_transform(observations.y)
            self.store.set_transformed_y(
                transformed, observation_version=observations.observation_version
            )
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
                force_full_refit=self._force_full_refit or reset_parameters,
                reset_parameters=reset_parameters,
            )
        self._model_observation_version = self.store.observation_version
        self._force_full_refit = False
        return report

    def _model_suggest(self, n_suggestions: int, fixed: FixedInput | None) -> CandidateBatch:
        assert self._surrogate is not None
        posterior = PosteriorEvaluator(
            self._surrogate,
            batch_size=self.config.runtime.acquisition_batch_size,
            cache=self.config.acquisition.posterior_cache,
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
            repaired = self.space.repair_dense(dense, fixed=fixed)
            encoded = self.space.encoded_from_dense(repaired, repair=False, fixed=fixed)
            return mace(encoded.continuous, encoded.categorical)

        search_objective = objective
        if self.config.runtime.enable_torch_compile:
            search_objective = torch.compile(objective, dynamic=True)
        search = TorchNSGA2(
            population_size=self.config.search.population_size,
            generations=self.config.search.generations,
            crossover_probability=self.config.search.crossover_probability,
            crossover_eta=self.config.search.crossover_eta,
            mutation_probability=self.config.search.mutation_probability,
            mutation_eta=self.config.search.mutation_eta,
            tournament_size=self.config.search.tournament_size,
            eliminate_duplicate_points=self.config.search.eliminate_duplicates,
        )
        specification = self._search_spec(fixed)
        previous = (
            self._previous_population if self.config.search.reuse_previous_population else None
        )
        with self.diagnostics.phase("suggest.search.total"):
            result = search.minimize(
                search_objective,
                space=specification,
                incumbents=self.space.to_dense(incumbent),
                initial_population=previous,
                generator=self.random.search,
            )
        self._last_search = result
        self._previous_population = result.population.detach()
        if self.config.search.keep_history:
            self._search_history.append(
                (result.population.detach().clone(), result.objectives.detach().clone())
            )
        pool = result.pareto_population
        if pool.shape[0] == 0:
            pool = result.population
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
        attempts = 0
        while retained < requested and attempts < 8:
            missing = requested - retained
            draw = self._draw_sobol(max(4, missing * 2), fixed)
            historical_unique = self.store.unique_mask(draw)
            keys = self.space.canonical_keys(draw)
            keep: list[int] = []
            for index, key in enumerate(keys):
                if bool(historical_unique[index]) and key not in seen:
                    keep.append(index)
                    seen.add(key)
                    retained += 1
                    if retained == requested:
                        break
            if keep:
                encoded_rows.append(draw.select(keep).encoded)
            attempts += 1
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

    def observe(self, candidates: CandidateBatch | EncodedBatch | object, y: object) -> int:
        """Append finite objective observations without doing eager GP work."""

        with self.diagnostics.phase("observe.total"):
            discarded_before = self.store.discarded_count
            retained = self.store.append(candidates, y)  # type: ignore[arg-type]
            self.diagnostics.increment("observe.received", retained)
            self.diagnostics.increment(
                "observe.discarded", self.store.discarded_count - discarded_before
            )
            return retained

    observe_new_data = observe

    def refit(self) -> FitReport | None:
        """Run an explicit full GP refit now when model-based data are available."""

        if len(self.store) < self.random_samples:
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

    def state_dict(self) -> dict[str, Any]:
        """Return only LeanHEBO-defined tensors and checkpoint-safe primitives."""

        observation_chunks = [
            {
                "continuous": encoded.continuous.detach().cpu(),
                "categorical": encoded.categorical.detach().cpu(),
                "y": outcomes.detach().cpu(),
            }
            for encoded, outcomes in zip(
                self.store.encoded_chunks, self.store.y_chunks, strict=True
            )
        ]
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "space": self.public_space.to_spec(),
            "space_fingerprint": self.space.fingerprint,
            "observations": observation_chunks,
            "duplicate_keys": [list(key) for key in self.store.key_snapshot()],
            "discarded_observations": self.store.discarded_count,
            "output_transform": self.output_transform.state_dict(),
            "surrogate": (None if self._surrogate is None else self._surrogate.state_dict()),
            "diagnostics": self.diagnostics.state_dict(),
            "random": self.random.state_dict(),
            "sobol_draw_count": self._sobol_draw_count,
            "observation_version": self.store.observation_version,
            "store_transform_version": self.store.transform_version,
            "model_observation_version": self._model_observation_version,
            "previous_population": (
                None
                if self._previous_population is None
                else self._previous_population.detach().cpu()
            ),
            "search_history": [
                {
                    "population": population.detach().cpu(),
                    "objectives": objectives.detach().cpu(),
                }
                for population, objectives in self._search_history
            ],
            "versions": {
                "torch": str(torch.__version__),
                "python": platform.python_version(),
                "leanhebo": _installed_version("leanhebo"),
                "gpytorch": _installed_version("gpytorch"),
                "numpy": _installed_version("numpy"),
                "schema": _STATE_SCHEMA_VERSION,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != _STATE_SCHEMA_VERSION:
            raise ValueError("unsupported LeanHEBO optimizer state schema")
        if state.get("space_fingerprint") != self.space.fingerprint:
            raise ValueError("checkpoint design space does not match this optimizer")
        saved_config = LeanHEBOConfig.from_dict(state["config"])
        if saved_config != self.config:
            raise ValueError("checkpoint configuration does not match this optimizer")
        saved_model_observation_version = int(state.get("model_observation_version", -1))
        # In older schema-v1 checkpoints each retained observation chunk advanced
        # the store version once. Infer that version instead of reviving a stale
        # surrogate as current merely because the explicit field is absent.
        saved_observation_version = int(
            state.get("observation_version", len(state["observations"]))
        )
        saved_store_transform_version = state["store_transform_version"]
        model_was_current = (
            state.get("surrogate") is not None
            and saved_model_observation_version == saved_observation_version
        )
        self.store.clear()
        for chunk in state["observations"]:
            encoded = EncodedBatch(chunk["continuous"], chunk["categorical"])
            self.store.append(encoded, chunk["y"])
        self.store.restore_keys(state["duplicate_keys"])
        discarded = state["discarded_observations"]
        if isinstance(discarded, bool) or not isinstance(discarded, int) or discarded < 0:
            raise ValueError("checkpoint discarded-observation count is invalid")
        self.store.discarded_count = discarded
        self.output_transform.load_state_dict(state["output_transform"])
        self.output_transform.to(device=self.device, dtype=self.dtype)
        surrogate_state = state.get("surrogate")
        if surrogate_state is not None:
            self._surrogate = self._make_surrogate()
            self._surrogate.load_state_dict(surrogate_state)
            if self.output_transform.fitted and len(self.store):
                self.store.set_transformed_y(self.output_transform.transform(self.store.y))
        else:
            self._surrogate = None
        self.store.restore_versions(
            saved_observation_version,
            saved_store_transform_version,
        )
        self.random.load_state_dict(dict(state["random"]))
        diagnostics_state = state["diagnostics"]
        if not isinstance(diagnostics_state, Mapping):
            raise TypeError("checkpoint diagnostics state is malformed")
        self.diagnostics.load_state_dict(diagnostics_state)
        self._sobol = self._new_sobol_engine()
        self._sobol_draw_count = int(state["sobol_draw_count"])
        if self._sobol_draw_count:
            self._sobol.fast_forward(  # type: ignore[no-untyped-call]
                self._sobol_draw_count
            )
        self._model_observation_version = (
            self.store.observation_version if model_was_current else -1
        )
        previous = state.get("previous_population")
        self._previous_population = (
            None
            if previous is None
            else torch.as_tensor(previous, device=self.device, dtype=self.dtype)
        )
        self._force_full_refit = False
        self._last_search = None
        history = state["search_history"]
        if not isinstance(history, list):
            raise TypeError("checkpoint search history is malformed")
        restored_history: list[tuple[torch.Tensor, torch.Tensor]] = []
        for item in history:
            if not isinstance(item, Mapping):
                raise TypeError("checkpoint search history entry is malformed")
            population = torch.as_tensor(item["population"], device=self.device, dtype=self.dtype)
            objectives = torch.as_tensor(item["objectives"], device=self.device, dtype=self.dtype)
            if population.ndim != 2 or population.shape[1] != self.space.dense_dimension:
                raise ValueError("checkpoint search population has an incompatible shape")
            if objectives.ndim != 2 or objectives.shape[0] != population.shape[0]:
                raise ValueError("checkpoint search objectives have an incompatible shape")
            restored_history.append((population.clone(), objectives.clone()))
        self._search_history = restored_history

    def save(self, path: str | Path) -> None:
        save_checkpoint(path, self.state_dict())

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
                state = dict(state)
                state["config"] = config.to_dict()
            space = Space.from_spec(state["space"])
            optimizer = cls(space, config=config)
            optimizer.load_state_dict(state)
            return optimizer
        except CheckpointError:
            raise
        except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as error:
            raise CheckpointError(
                f"checkpoint payload is malformed or incompatible: {error}"
            ) from error


def _installed_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"
