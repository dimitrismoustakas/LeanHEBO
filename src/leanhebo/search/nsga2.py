# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""A narrow, tensor-native NSGA-II minimizer for MACE and analytic objectives."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from leanhebo.search.duplicates import eliminate_duplicates
from leanhebo.search.operators import (
    _validate_generator,
    binary_tournament,
    mixed_variable_crossover,
    mutate_population,
)
from leanhebo.search.repair import MixedVariableSpec, repair_population
from leanhebo.search.sorting import crowding_distance, non_dominated_sort
from leanhebo.search.survival import elitist_survival

Objective = Callable[[Tensor], Tensor]
TensorLike = Tensor | Sequence[float]


@dataclass(frozen=True, slots=True)
class NSGA2Result:
    """Final dense population and its NSGA-II metrics."""

    population: Tensor
    objectives: Tensor
    ranks: Tensor
    crowding: Tensor
    generations: int

    @property
    def pareto_mask(self) -> Tensor:
        """Mask selecting the rank-zero front."""

        return self.ranks == 0

    @property
    def pareto_indices(self) -> Tensor:
        """Indices of the rank-zero front."""

        return torch.nonzero(self.pareto_mask, as_tuple=False).flatten()

    @property
    def pareto_population(self) -> Tensor:
        """Decision vectors on the rank-zero front."""

        return self.population[self.pareto_mask]

    @property
    def pareto_objectives(self) -> Tensor:
        """Objective vectors on the rank-zero front."""

        return self.objectives[self.pareto_mask]

    @property
    def x(self) -> Tensor:
        """Short compatibility alias for the final population."""

        return self.population

    @property
    def f(self) -> Tensor:
        """Short compatibility alias for the final objectives."""

        return self.objectives


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

        integer = self._spec.integer_mask & ~self._spec.fixed_mask
        if bool(integer.any()):
            steps = self._spec.steps[integer]
            cardinality = (
                torch.floor((self._spec.upper[integer] - self._spec.lower[integer]) / steps) + 1
            )
            codes = torch.floor(unit[:, integer] * cardinality).clamp_max(cardinality - 1)
            population[:, integer] = self._spec.lower[integer] + codes * steps

        categorical = self._spec.categorical_mask & ~self._spec.fixed_mask
        if bool(categorical.any()):
            cardinality = self._spec.upper[categorical] - self._spec.lower[categorical] + 1
            codes = torch.floor(unit[:, categorical] * cardinality).clamp_max(cardinality - 1)
            population[:, categorical] = self._spec.lower[categorical] + codes
        return repair_population(population, self._spec)


def sobol_population(
    spec: MixedVariableSpec,
    population_size: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw and repair a mixed-variable Sobol population.

    A generator controls the scramble seed and must live on the same device as ``spec``. The Sobol
    engine itself is CPU-based; CUDA receives only the completed dense unit-cube draw.
    """

    if population_size < 0:
        raise ValueError("population_size must be non-negative")
    _validate_generator(generator, spec.lower.device)
    return _SobolPopulationSampler(spec, generator).draw(population_size)


# Explicit alternative spelling for discoverability.
initialize_sobol_population = sobol_population


class TorchNSGA2:
    """Fixed-generation NSGA-II over a dense mixed-variable Torch population.

    The objective must accept a tensor shaped ``[n, d]`` and return minimization values shaped
    ``[n, m]`` (or ``[n]`` for a single objective). Bounds or a :class:`MixedVariableSpec` can be
    supplied once to the constructor or to each :meth:`minimize` call. No NumPy or tabular objects
    enter the evolutionary loop.
    """

    def __init__(
        self,
        space: MixedVariableSpec | None = None,
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
        eliminate_duplicates: bool | None = None,
        duplicate_tolerance: float = 0.0,
        max_duplicate_retries: int = 4,
        pop_size: int | None = None,
        n_generations: int | None = None,
    ) -> None:
        """Configure search work and operators.

        ``pop_size`` and ``n_generations`` are accepted as familiar NSGA-II aliases. The explicit
        ``population_size`` and ``generations`` spellings are preferred.
        """

        if pop_size is not None:
            if population_size != 100:
                raise ValueError("specify only one of population_size and pop_size")
            population_size = pop_size
        if n_generations is not None:
            if generations != 100:
                raise ValueError("specify only one of generations and n_generations")
            generations = n_generations
        if eliminate_duplicates is not None:
            if eliminate_duplicate_points is not True:
                raise ValueError(
                    "specify only one of eliminate_duplicate_points and eliminate_duplicates"
                )
            eliminate_duplicate_points = eliminate_duplicates
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
        if not math.isfinite(duplicate_tolerance) or duplicate_tolerance < 0:
            raise ValueError("duplicate_tolerance must be non-negative and finite")
        if (
            isinstance(max_duplicate_retries, bool)
            or not isinstance(max_duplicate_retries, int)
            or max_duplicate_retries < 0
        ):
            raise ValueError("max_duplicate_retries must be non-negative")

        self.space = space
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.crossover_dimension_probability = crossover_dimension_probability
        self.crossover_eta = crossover_eta
        self.mutation_probability = mutation_probability
        self.mutation_eta = mutation_eta
        self.tournament_size = tournament_size
        self.eliminate_duplicate_points = eliminate_duplicate_points
        self.duplicate_tolerance = duplicate_tolerance
        self.max_duplicate_retries = max_duplicate_retries

    def _resolve_space(
        self,
        lower: TensorLike | MixedVariableSpec | None,
        upper: TensorLike | None,
        *,
        space: MixedVariableSpec | None,
        integer_mask: Tensor | None,
        categorical_mask: Tensor | None,
        steps: Tensor | None,
        fixed_mask: Tensor | None,
        fixed_values: Tensor | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> MixedVariableSpec:
        candidates = [
            space is not None,
            isinstance(lower, MixedVariableSpec),
            self.space is not None,
        ]
        if sum(candidates) > 1:
            raise ValueError("provide a search specification in only one place")
        if space is not None:
            resolved = space
        elif isinstance(lower, MixedVariableSpec):
            if upper is not None:
                raise ValueError("upper must be omitted when lower is a MixedVariableSpec")
            resolved = lower
        elif self.space is not None:
            if lower is not None or upper is not None:
                raise ValueError("bounds cannot override the constructor search specification")
            resolved = self.space
        else:
            if lower is None or upper is None:
                raise ValueError(
                    "provide either a MixedVariableSpec or both lower and upper bounds"
                )
            if device is None and isinstance(lower, Tensor):
                target_device = lower.device
            else:
                target_device = torch.device("cpu") if device is None else torch.device(device)
            if dtype is None and isinstance(lower, Tensor) and lower.is_floating_point():
                target_dtype = lower.dtype
            else:
                target_dtype = torch.get_default_dtype() if dtype is None else dtype
            resolved = MixedVariableSpec(
                lower=torch.as_tensor(lower, device=target_device, dtype=target_dtype),
                upper=torch.as_tensor(upper, device=target_device, dtype=target_dtype),
                integer_mask=integer_mask,
                categorical_mask=categorical_mask,
                steps=steps,
                fixed_mask=fixed_mask,
                fixed_values=fixed_values,
            )
        return resolved.to(device=device, dtype=dtype)

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
            seeded = eliminate_duplicates(
                seeded,
                spec=spec,
                atol=self.duplicate_tolerance,
            )
        population = seeded[: self.population_size]

        attempts = 0
        while population.shape[0] < self.population_size:
            remaining = self.population_size - population.shape[0]
            draw_count = remaining if not self.eliminate_duplicate_points else max(remaining * 2, 4)
            candidates = sampler.draw(draw_count)
            if self.eliminate_duplicate_points:
                candidates = eliminate_duplicates(
                    candidates,
                    existing=population,
                    spec=spec,
                    atol=self.duplicate_tolerance,
                )
            take = min(remaining, candidates.shape[0])
            if take:
                population = torch.cat((population, candidates[:take]), dim=0)
            attempts += 1
            if attempts > self.max_duplicate_retries and population.shape[0] < self.population_size:
                break
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
        with torch.no_grad():
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
        objectives: Tensor,
        spec: MixedVariableSpec,
        sampler: _SobolPopulationSampler,
        generator: torch.Generator | None,
    ) -> Tensor:
        target = population.shape[0]
        ranks = non_dominated_sort(objectives)
        crowding = crowding_distance(objectives, ranks)
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
            candidates = eliminate_duplicates(
                candidates,
                existing=existing,
                spec=spec,
                atol=self.duplicate_tolerance,
            )
            offspring = torch.cat((offspring, candidates[:remaining]), dim=0)

        remaining = target - offspring.shape[0]
        if remaining > 0:
            # Saturated discrete spaces can make variation repeatedly reproduce a parent. A Sobol
            # refill preserves dense execution and either finds unseen canonical points or proves
            # (within the configured retries) that the available population is exhausted.
            candidates = sampler.draw(max(remaining * 2, 4))
            candidates = eliminate_duplicates(
                candidates,
                existing=torch.cat((population, offspring), dim=0),
                spec=spec,
                atol=self.duplicate_tolerance,
            )
            offspring = torch.cat((offspring, candidates[:remaining]), dim=0)
        return offspring

    def minimize(
        self,
        objective: Objective,
        lower: TensorLike | MixedVariableSpec | None = None,
        upper: TensorLike | None = None,
        *,
        space: MixedVariableSpec | None = None,
        integer_mask: Tensor | None = None,
        categorical_mask: Tensor | None = None,
        steps: Tensor | None = None,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
        incumbent: Tensor | None = None,
        incumbents: Tensor | None = None,
        initial_population: Tensor | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> NSGA2Result:
        """Minimize a batched Torch objective for the configured number of generations."""

        if incumbent is not None:
            if incumbents is not None:
                raise ValueError("specify only one of incumbent and incumbents")
            incumbents = incumbent
        spec = self._resolve_space(
            lower,
            upper,
            space=space,
            integer_mask=integer_mask,
            categorical_mask=categorical_mask,
            steps=steps,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
            device=device,
            dtype=dtype,
        )
        _validate_generator(generator, spec.lower.device)
        sampler = _SobolPopulationSampler(spec, generator)
        population = self._initialize(
            sampler,
            spec,
            incumbents=incumbents,
            initial_population=initial_population,
        )
        objectives = self._evaluate(objective, population)
        completed_generations = 0

        for _ in range(self.generations):
            offspring = self._make_offspring(
                population,
                objectives,
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
            combined_population = torch.cat((population, offspring), dim=0)
            combined_objectives = torch.cat((objectives, offspring_objectives), dim=0)
            survival = elitist_survival(
                combined_population,
                combined_objectives,
                population.shape[0],
                spec=spec,
                eliminate_duplicate_points=self.eliminate_duplicate_points,
                duplicate_tolerance=self.duplicate_tolerance,
            )
            population = survival.population
            objectives = survival.objectives
            completed_generations += 1

        ranks = non_dominated_sort(objectives)
        crowding = crowding_distance(objectives, ranks)
        return NSGA2Result(
            population=population,
            objectives=objectives,
            ranks=ranks,
            crowding=crowding,
            generations=completed_generations,
        )
