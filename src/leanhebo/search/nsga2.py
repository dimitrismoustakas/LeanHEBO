# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""A narrow, tensor-native NSGA-II minimizer for MACE and analytic objectives."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from leanhebo.search.duplicates import _exact_duplicate_mask
from leanhebo.search.operators import (
    _validate_generator,
    binary_tournament,
    mixed_variable_crossover,
    mutate_population,
)
from leanhebo.search.repair import (
    MixedVariableSpec,
    _repair_population_unchecked,
    repair_population,
)
from leanhebo.search.sorting import (
    _front_crowding,
    _non_dominated_sort_unchecked,
    _ranked_crowding,
)
from leanhebo.search.survival import _select_survivors_unchecked

Objective = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class NSGA2Result:
    """Final dense population and its NSGA-II metrics."""

    population: Tensor
    objectives: Tensor
    ranks: Tensor
    crowding: Tensor
    generations: int
    objective_calls: int
    candidate_evaluations: int

    @property
    def pareto_mask(self) -> Tensor:
        """Mask selecting the rank-zero front."""

        return self.ranks == 0

    @property
    def pareto_population(self) -> Tensor:
        """Decision vectors on the rank-zero front."""

        return self.population[self.pareto_mask]


class _SobolPopulationSampler:
    def __init__(
        self,
        spec: MixedVariableSpec,
        generator: torch.Generator | None,
    ) -> None:
        seed: int | None = None
        if generator is not None:
            seed = int(
                torch.randint(
                    0,
                    2**31 - 1,
                    (),
                    device=spec.lower.device,
                    generator=generator,
                ).item()
            )
        self._spec = spec
        self._engine = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
            spec.dimension,
            scramble=True,
            seed=seed,
        )

    def draw(self, count: int) -> Tensor:
        if count < 0:
            raise ValueError("Sobol draw count must be non-negative")
        if count == 0:
            return self._spec.lower.new_empty((0, self._spec.dimension))
        draw_dtype = (
            self._spec.lower.dtype
            if self._spec.lower.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
        unit = self._engine.draw(count, dtype=draw_dtype).to(
            device=self._spec.lower.device,
            dtype=self._spec.lower.dtype,
        )
        population = self._spec.lower + unit * (self._spec.upper - self._spec.lower)

        integer = self._spec.mutable_integer_mask
        if self._spec.has_integer:
            steps = self._spec.steps[integer]
            cardinality = (
                torch.floor((self._spec.upper[integer] - self._spec.lower[integer]) / steps) + 1
            )
            codes = torch.floor(unit[:, integer] * cardinality).clamp_max(cardinality - 1)
            population[:, integer] = self._spec.lower[integer] + codes * steps

        categorical = self._spec.mutable_categorical_mask
        if self._spec.has_categorical:
            cardinality = self._spec.upper[categorical] - self._spec.lower[categorical] + 1
            codes = torch.floor(unit[:, categorical] * cardinality).clamp_max(cardinality - 1)
            population[:, categorical] = self._spec.lower[categorical] + codes
        return _repair_population_unchecked(population, self._spec)


def _discrete_lattice_completion(
    spec: MixedVariableSpec,
    count: int,
    *,
    existing: Tensor,
) -> Tensor:
    """Enumerate deterministic unseen rows for a fully discrete search lattice."""

    if count <= 0 or not bool((spec.integer_mask | spec.categorical_mask | spec.fixed_mask).all()):
        return spec.lower.new_empty((0, spec.dimension))

    cardinalities: list[int] = []
    for dimension in range(spec.dimension):
        if bool(spec.fixed_mask[dimension]):
            cardinalities.append(1)
        elif bool(spec.integer_mask[dimension]):
            span = (spec.upper[dimension] - spec.lower[dimension]) / spec.steps[dimension]
            cardinalities.append(math.floor(float(span.item())) + 1)
        else:
            span = spec.upper[dimension] - spec.lower[dimension]
            cardinalities.append(round(float(span.item())) + 1)

    total = math.prod(cardinalities)
    completed = spec.lower.new_empty((0, spec.dimension))
    chunk_size = max(64, min(4096, count * 4))
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        ordinals = torch.arange(
            start,
            stop,
            dtype=torch.int64,
            device=spec.lower.device,
        )
        remainder = ordinals
        candidates = spec.lower.expand(stop - start, -1).clone()
        for dimension in range(spec.dimension - 1, -1, -1):
            cardinality = cardinalities[dimension]
            codes = remainder.remainder(cardinality)
            remainder = torch.div(remainder, cardinality, rounding_mode="floor")
            if bool(spec.fixed_mask[dimension]):
                candidates[:, dimension] = spec.fixed_values[dimension]
            elif bool(spec.integer_mask[dimension]):
                candidates[:, dimension] += codes.to(spec.lower.dtype) * spec.steps[dimension]
            else:
                candidates[:, dimension] += codes.to(spec.lower.dtype)
        candidates = _repair_population_unchecked(candidates, spec)
        seen = torch.cat((existing, completed), dim=0)
        candidates = candidates[~_exact_duplicate_mask(candidates, seen)]
        completed = torch.cat((completed, candidates[: count - completed.shape[0]]), dim=0)
        if completed.shape[0] == count:
            break
    return completed


class TorchNSGA2:
    """Fixed-generation NSGA-II over a dense mixed-variable Torch population.

    The objective must accept a tensor shaped ``[n, d]`` and return minimization values shaped
    ``[n, m]`` (or ``[n]`` for a single objective). No NumPy or tabular objects enter the
    evolutionary loop.
    """

    def __init__(
        self,
        space: MixedVariableSpec,
        *,
        population_size: int = 100,
        generations: int = 100,
        crossover_probability: float = 0.9,
        crossover_dimension_probability: float = 0.5,
        crossover_eta: float = 15.0,
        mutation_probability: float | None = None,
        mutation_eta: float = 20.0,
        tournament_size: int = 2,
        eliminate_duplicate_points: bool = True,
        max_duplicate_retries: int = 4,
    ) -> None:
        """Configure search work and operators."""
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size <= 0
        ):
            raise ValueError("population_size must be positive")
        if isinstance(generations, bool) or not isinstance(generations, int) or generations < 0:
            raise ValueError("generations must be non-negative")
        for name, value in (
            ("crossover_probability", crossover_probability),
            ("crossover_dimension_probability", crossover_dimension_probability),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if mutation_probability is not None and (
            not math.isfinite(mutation_probability) or not 0 <= mutation_probability <= 1
        ):
            raise ValueError("mutation_probability must lie in [0, 1]")
        if not math.isfinite(crossover_eta) or crossover_eta <= 0:
            raise ValueError("crossover_eta must be positive and finite")
        if not math.isfinite(mutation_eta) or mutation_eta <= 0:
            raise ValueError("mutation_eta must be positive and finite")
        if (
            isinstance(tournament_size, bool)
            or not isinstance(tournament_size, int)
            or tournament_size < 2
        ):
            raise ValueError("tournament_size must be at least two")
        if (
            isinstance(max_duplicate_retries, bool)
            or not isinstance(max_duplicate_retries, int)
            or max_duplicate_retries < 0
        ):
            raise ValueError("max_duplicate_retries must be non-negative")

        self._spec = space
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.crossover_dimension_probability = crossover_dimension_probability
        self.crossover_eta = crossover_eta
        self.mutation_probability = mutation_probability
        self.mutation_eta = mutation_eta
        self.tournament_size = tournament_size
        self.eliminate_duplicate_points = eliminate_duplicate_points
        self.max_duplicate_retries = max_duplicate_retries

    @staticmethod
    def _as_seed_population(
        value: Tensor | None,
        *,
        spec: MixedVariableSpec,
        name: str,
    ) -> Tensor:
        if value is None:
            return spec.lower.new_empty((0, spec.dimension))
        tensor = torch.as_tensor(value, device=spec.lower.device, dtype=spec.lower.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != spec.dimension:
            raise ValueError(f"{name} must have shape [n, {spec.dimension}]")
        return repair_population(tensor, spec)

    def _initialize(
        self,
        sampler: _SobolPopulationSampler,
        spec: MixedVariableSpec,
        *,
        incumbents: Tensor | None,
        initial_population: Tensor | None,
    ) -> Tensor:
        incumbent_rows = self._as_seed_population(
            incumbents,
            spec=spec,
            name="incumbents",
        )
        initial_rows = self._as_seed_population(
            initial_population,
            spec=spec,
            name="initial_population",
        )
        seeded = torch.cat((incumbent_rows, initial_rows), dim=0)
        if self.eliminate_duplicate_points and seeded.shape[0]:
            seeded = seeded[~_exact_duplicate_mask(seeded)]
        population = seeded[: self.population_size]

        attempts = 0
        while population.shape[0] < self.population_size:
            remaining = self.population_size - population.shape[0]
            draw_count = remaining if not self.eliminate_duplicate_points else max(remaining * 2, 4)
            candidates = sampler.draw(draw_count)
            if self.eliminate_duplicate_points:
                candidates = candidates[~_exact_duplicate_mask(candidates, population)]
            take = min(remaining, candidates.shape[0])
            if take:
                population = torch.cat((population, candidates[:take]), dim=0)
            attempts += 1
            if attempts > self.max_duplicate_retries and population.shape[0] < self.population_size:
                break
        if self.eliminate_duplicate_points and population.shape[0] < self.population_size:
            completion = _discrete_lattice_completion(
                spec,
                self.population_size - population.shape[0],
                existing=population,
            )
            population = torch.cat((population, completion), dim=0)
        if population.shape[0] == 0:  # A positive Sobol draw always yields at least one row.
            raise RuntimeError("failed to initialize a non-empty population")
        return population

    @staticmethod
    def _evaluate(
        objective: Objective,
        population: Tensor,
        *,
        expected_objectives: int | None = None,
    ) -> Tensor:
        with torch.inference_mode():
            values = objective(population)
        if not isinstance(values, Tensor):
            raise TypeError("objective must return a torch.Tensor")
        if values.ndim == 1:
            values = values.unsqueeze(1)
        if values.ndim != 2 or values.shape[0] != population.shape[0]:
            raise ValueError("objective must return shape [population, objectives]")
        if values.shape[1] == 0:
            raise ValueError("objective must return at least one objective column")
        if expected_objectives is not None and values.shape[1] != expected_objectives:
            raise ValueError("objective column count changed between evaluations")
        if values.device != population.device:
            raise ValueError("objective values and population must be on the same device")
        if not values.is_floating_point():
            raise TypeError("objective values must use a floating-point dtype")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("objective returned non-finite values")
        return values.detach()

    def _offspring_batch(
        self,
        population: Tensor,
        ranks: Tensor,
        crowding: Tensor,
        spec: MixedVariableSpec,
        count: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        pair_count = math.ceil(count / 2)
        parent_indices = binary_tournament(
            ranks,
            crowding,
            pair_count * 2,
            generator=generator,
            tournament_size=self.tournament_size,
        )
        parent_a = population[parent_indices[0::2]]
        parent_b = population[parent_indices[1::2]]
        child_a, child_b = mixed_variable_crossover(
            parent_a,
            parent_b,
            spec,
            probability=self.crossover_probability,
            dimension_probability=self.crossover_dimension_probability,
            eta=self.crossover_eta,
            generator=generator,
        )
        children = torch.stack((child_a, child_b), dim=1).reshape(-1, spec.dimension)[:count]
        return mutate_population(
            children,
            spec,
            probability=self.mutation_probability,
            eta=self.mutation_eta,
            generator=generator,
        )

    def _make_offspring(
        self,
        population: Tensor,
        ranks: Tensor,
        crowding: Tensor,
        spec: MixedVariableSpec,
        sampler: _SobolPopulationSampler,
        generator: torch.Generator | None,
    ) -> Tensor:
        target = population.shape[0]
        if not self.eliminate_duplicate_points:
            return self._offspring_batch(
                population,
                ranks,
                crowding,
                spec,
                target,
                generator,
            )

        offspring = population.new_empty((0, spec.dimension))
        for _ in range(self.max_duplicate_retries + 1):
            remaining = target - offspring.shape[0]
            if remaining <= 0:
                break
            candidates = self._offspring_batch(
                population,
                ranks,
                crowding,
                spec,
                max(remaining * 2, 2),
                generator,
            )
            existing = torch.cat((population, offspring), dim=0)
            candidates = candidates[~_exact_duplicate_mask(candidates, existing)]
            offspring = torch.cat((offspring, candidates[:remaining]), dim=0)

        remaining = target - offspring.shape[0]
        if remaining > 0:
            # Give variation one diverse Sobol refill before the exact discrete-lattice completion
            # below. Spaces with a continuous axis remain best-effort because they are not finite.
            candidates = sampler.draw(max(remaining * 2, 4))
            seen = torch.cat((population, offspring), dim=0)
            candidates = candidates[~_exact_duplicate_mask(candidates, seen)]
            offspring = torch.cat((offspring, candidates[:remaining]), dim=0)
        remaining = target - offspring.shape[0]
        if remaining > 0:
            completion = _discrete_lattice_completion(
                spec,
                remaining,
                existing=torch.cat((population, offspring), dim=0),
            )
            offspring = torch.cat((offspring, completion), dim=0)
        return offspring

    def minimize(
        self,
        objective: Objective,
        *,
        incumbents: Tensor | None = None,
        initial_population: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> NSGA2Result:
        """Minimize a batched Torch objective for the configured number of generations."""

        spec = self._spec
        _validate_generator(generator, spec.lower.device)
        sampler = _SobolPopulationSampler(spec, generator)
        population = self._initialize(
            sampler,
            spec,
            incumbents=incumbents,
            initial_population=initial_population,
        )
        objectives = self._evaluate(objective, population)
        ranks, _ = _non_dominated_sort_unchecked(objectives)
        crowding = _ranked_crowding(objectives, ranks)
        completed_generations = 0
        objective_calls = 1
        candidate_evaluations = population.shape[0]

        for _ in range(self.generations):
            offspring = self._make_offspring(
                population,
                ranks,
                crowding,
                spec,
                sampler,
                generator,
            )
            if offspring.shape[0] == 0:
                break
            offspring_objectives = self._evaluate(
                objective,
                offspring,
                expected_objectives=objectives.shape[1],
            )
            objective_calls += 1
            candidate_evaluations += offspring.shape[0]
            combined_population = torch.cat((population, offspring), dim=0)
            combined_objectives = torch.cat((objectives, offspring_objectives), dim=0)
            # The pool is duplicate-free by construction: the initial population is unique,
            # survivors remain a subset, and ``_make_offspring`` removes matches both within
            # the batch and against the parents.
            selection = _select_survivors_unchecked(combined_objectives, population.shape[0])
            survivors = selection.indices
            population = combined_population[survivors]
            objectives = combined_objectives[survivors]
            ranks = selection.ranks[survivors]
            crowding = selection.crowding[survivors].clone()
            # Every front before the final retained front survives intact, so its crowding values
            # remain valid. Only the possibly truncated final front needs to be refreshed.
            final_front = ranks == ranks.max()
            crowding[final_front] = _front_crowding(objectives[final_front])
            completed_generations += 1

        return NSGA2Result(
            population=population,
            objectives=objectives,
            ranks=ranks,
            crowding=crowding,
            generations=completed_generations,
            objective_calls=objective_calls,
            candidate_evaluations=candidate_evaluations,
        )
