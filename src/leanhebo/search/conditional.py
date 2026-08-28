# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Conditional-space NSGA-II with exact semantic duplicate handling."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from leanhebo.search.conditional_operators import conditional_mutation
from leanhebo.search.duplicates import _exact_duplicate_mask
from leanhebo.search.nsga2 import TorchNSGA2, _SobolPopulationSampler
from leanhebo.search.operators import binary_tournament, mixed_variable_crossover
from leanhebo.search.repair import MixedVariableSpec, repair_population


class ConditionalSearchSemantics(Protocol):
    """Conditional-space operations required by the evolutionary search.

    Contextual fixed assignments belong to the semantics instance. All returned tensors stay on
    the population device so activity and duplicate handling remain inside the tensor search loop.
    """

    def activity_mask(self, population: Tensor) -> Tensor:
        """Return a boolean ``[population, dense dimensions]`` activity mask."""

        ...

    def semantic_keys(self, population: Tensor) -> Tensor:
        """Return exact, canonical ``[population, key dimensions]`` keys."""

        ...

    def finite_completion(self, count: int, *, existing: Tensor) -> Tensor:
        """Return up to ``count`` contextual finite-space rows not in ``existing``."""

        ...


def _validate_semantic_keys(keys: Tensor, population: Tensor, *, name: str) -> None:
    if not isinstance(keys, Tensor):
        raise TypeError(f"{name} must return a torch.Tensor")
    if keys.ndim != 2 or keys.shape[0] != population.shape[0]:
        raise ValueError(f"{name} must return shape [population, key dimensions]")
    if keys.device != population.device:
        raise ValueError(f"{name} must return keys on the population device")
    if keys.is_complex():
        raise TypeError(f"{name} must return real-valued keys")


def _exact_key_duplicate_mask(keys: Tensor, existing_keys: Tensor | None = None) -> Tensor:
    """Mark exact duplicate key rows while preserving first-occurrence order."""

    if existing_keys is not None and existing_keys.shape[0]:
        if existing_keys.ndim != 2 or existing_keys.shape[1] != keys.shape[1]:
            raise ValueError("existing semantic keys have incompatible dimensions")
        if existing_keys.device != keys.device or existing_keys.dtype != keys.dtype:
            raise ValueError("existing semantic keys must share key device and dtype")
    return _exact_duplicate_mask(keys, existing_keys)


def _semantic_keys(population: Tensor, semantics: ConditionalSearchSemantics) -> Tensor:
    keys = semantics.semantic_keys(population)
    _validate_semantic_keys(keys, population, name="semantic_keys")
    return keys


def _eliminate_keyed(
    population: Tensor,
    keys: Tensor,
    *,
    existing_keys: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    keep = ~_exact_key_duplicate_mask(keys, existing_keys)
    return population[keep], keys[keep]


def semantic_duplicate_mask(
    population: Tensor,
    semantics: ConditionalSearchSemantics,
    *,
    existing: Tensor | None = None,
) -> Tensor:
    """Mark exact semantic duplicates using on-device canonical keys."""

    keys = _semantic_keys(population, semantics)
    existing_keys: Tensor | None = None
    if existing is not None:
        if existing.ndim != 2 or existing.shape[1] != population.shape[1]:
            raise ValueError("existing and population must have the same dense dimensions")
        if existing.device != population.device or existing.dtype != population.dtype:
            raise ValueError("existing and population must share device and dtype")
        existing_keys = _semantic_keys(existing, semantics)
    return _exact_key_duplicate_mask(keys, existing_keys)


def eliminate_semantic_duplicates(
    population: Tensor,
    semantics: ConditionalSearchSemantics,
    *,
    existing: Tensor | None = None,
) -> Tensor:
    """Keep the first dense representative of every exact semantic configuration."""

    keep = ~semantic_duplicate_mask(population, semantics, existing=existing)
    return population[keep]


class ConditionalTorchNSGA2(TorchNSGA2):
    """Dense latent-gene NSGA-II whose equality and mutation follow conditional semantics."""

    def __init__(
        self,
        space: MixedVariableSpec,
        semantics: ConditionalSearchSemantics,
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
        super().__init__(
            space,
            population_size=population_size,
            generations=generations,
            crossover_probability=crossover_probability,
            crossover_dimension_probability=crossover_dimension_probability,
            crossover_eta=crossover_eta,
            mutation_probability=mutation_probability,
            mutation_eta=mutation_eta,
            tournament_size=tournament_size,
            eliminate_duplicate_points=eliminate_duplicate_points,
            max_duplicate_retries=max_duplicate_retries,
        )
        self.semantics = semantics

    def _completion(self, count: int, *, existing: Tensor) -> Tensor:
        if count <= 0:
            return existing.new_empty((0, self._spec.dimension))
        completion = self.semantics.finite_completion(count, existing=existing)
        if not isinstance(completion, Tensor):
            raise TypeError("finite_completion must return a torch.Tensor")
        if completion.ndim != 2 or completion.shape[1] != self._spec.dimension:
            raise ValueError(f"finite_completion must return shape [n, {self._spec.dimension}]")
        if completion.device != existing.device or completion.dtype != existing.dtype:
            raise ValueError("finite_completion must share population device and dtype")
        return repair_population(completion, self._spec)[:count]

    def _initialize(
        self,
        sampler: _SobolPopulationSampler,
        spec: MixedVariableSpec,
        *,
        incumbents: Tensor | None,
        initial_population: Tensor | None,
    ) -> Tensor:
        incumbent_rows = self._as_seed_population(incumbents, spec=spec, name="incumbents")
        initial_rows = self._as_seed_population(
            initial_population,
            spec=spec,
            name="initial_population",
        )
        seeded = torch.cat((incumbent_rows, initial_rows), dim=0)
        population_keys: Tensor | None = None
        if self.eliminate_duplicate_points:
            seeded_keys = _semantic_keys(seeded, self.semantics)
            seeded, seeded_keys = _eliminate_keyed(seeded, seeded_keys)
            population_keys = seeded_keys[: self.population_size]
        population = seeded[: self.population_size]

        attempts = 0
        while population.shape[0] < self.population_size:
            remaining = self.population_size - population.shape[0]
            draw_count = remaining if not self.eliminate_duplicate_points else max(remaining * 2, 4)
            candidates = sampler.draw(draw_count)
            if self.eliminate_duplicate_points:
                assert population_keys is not None
                candidate_keys = _semantic_keys(candidates, self.semantics)
                candidates, candidate_keys = _eliminate_keyed(
                    candidates,
                    candidate_keys,
                    existing_keys=population_keys,
                )
            take = min(remaining, candidates.shape[0])
            if take:
                population = torch.cat((population, candidates[:take]), dim=0)
                if self.eliminate_duplicate_points:
                    assert population_keys is not None
                    population_keys = torch.cat(
                        (population_keys, candidate_keys[:take]),
                        dim=0,
                    )
            attempts += 1
            if attempts > self.max_duplicate_retries and population.shape[0] < self.population_size:
                break

        if self.eliminate_duplicate_points and population.shape[0] < self.population_size:
            completion = self._completion(
                self.population_size - population.shape[0],
                existing=population,
            )
            population = torch.cat((population, completion), dim=0)
        if population.shape[0] == 0:
            raise RuntimeError("failed to initialize a non-empty population")
        return population

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
        return conditional_mutation(
            children,
            spec,
            self.semantics,
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
        population_keys = _semantic_keys(population, self.semantics)
        offspring_keys = population_keys.new_empty((0, population_keys.shape[1]))
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
            candidate_keys = _semantic_keys(candidates, self.semantics)
            candidates, candidate_keys = _eliminate_keyed(
                candidates,
                candidate_keys,
                existing_keys=torch.cat((population_keys, offspring_keys), dim=0),
            )
            take = min(remaining, candidates.shape[0])
            offspring = torch.cat((offspring, candidates[:take]), dim=0)
            offspring_keys = torch.cat((offspring_keys, candidate_keys[:take]), dim=0)

        remaining = target - offspring.shape[0]
        if remaining > 0:
            candidates = sampler.draw(max(remaining * 2, 4))
            candidate_keys = _semantic_keys(candidates, self.semantics)
            candidates, candidate_keys = _eliminate_keyed(
                candidates,
                candidate_keys,
                existing_keys=torch.cat((population_keys, offspring_keys), dim=0),
            )
            take = min(remaining, candidates.shape[0])
            offspring = torch.cat((offspring, candidates[:take]), dim=0)
            offspring_keys = torch.cat((offspring_keys, candidate_keys[:take]), dim=0)
        remaining = target - offspring.shape[0]
        if remaining > 0:
            completion = self._completion(
                remaining,
                existing=torch.cat((population, offspring), dim=0),
            )
            offspring = torch.cat((offspring, completion), dim=0)
        return offspring
