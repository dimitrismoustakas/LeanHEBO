# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Evolutionary operators whose mutation follows conditional activity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leanhebo.search.repair import MixedVariableSpec, _repair_population_unchecked

if TYPE_CHECKING:
    from leanhebo.search.conditional import ConditionalSearchSemantics


def _validated_activity_mask(
    population: Tensor,
    semantics: ConditionalSearchSemantics,
) -> Tensor:
    active = semantics.activity_mask(population)
    if not isinstance(active, Tensor):
        raise TypeError("activity_mask must return a torch.Tensor")
    if active.shape != population.shape:
        raise ValueError("activity_mask must return one flag per population coordinate")
    if active.dtype is not torch.bool:
        raise TypeError("activity_mask must return a boolean tensor")
    if active.device != population.device:
        raise ValueError("activity_mask must return a mask on the population device")
    return active


def conditional_mutation(
    population: Tensor,
    spec: MixedVariableSpec,
    semantics: ConditionalSearchSemantics,
    *,
    probability: float | None = None,
    eta: float = 20.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Mutate active, nonfixed genes in an already repaired population."""

    if population.shape[0] == 0 or probability == 0:
        return population.clone()
    active = _validated_activity_mask(population, semantics)
    categorical_counts = torch.round(spec.upper - spec.lower + 1.0).to(torch.long)
    mutable_numeric = spec.mutable_numeric_mask & (spec.upper > spec.lower)
    mutable_categorical = spec.mutable_categorical_mask & (categorical_counts > 1)
    mutable = active & (mutable_numeric | mutable_categorical)

    if probability is None:
        mutable_counts = mutable.sum(dim=1, keepdim=True)
        row_probability = torch.where(
            mutable_counts > 0,
            mutable_counts.reciprocal().to(dtype=population.dtype),
            torch.zeros_like(mutable_counts, dtype=population.dtype),
        )
    else:
        row_probability = population.new_full((population.shape[0], 1), probability)

    mutate = (
        torch.rand(population.shape, device=population.device, generator=generator)
        < row_probability
    ) & mutable

    span = spec.upper - spec.lower
    safe_span = span.clamp_min(torch.finfo(population.dtype).eps)
    delta_lower = (population - spec.lower) / safe_span
    delta_upper = (spec.upper - population) / safe_span
    random = torch.rand(
        population.shape,
        dtype=population.dtype,
        device=population.device,
        generator=generator,
    )
    mutation_power = 1.0 / (eta + 1.0)
    lower_base = 2.0 * random + (1.0 - 2.0 * random) * (1.0 - delta_lower).pow(eta + 1.0)
    lower_delta = lower_base.clamp_min(0).pow(mutation_power) - 1.0
    upper_base = 2.0 * (1.0 - random) + 2.0 * (random - 0.5) * (1.0 - delta_upper).pow(eta + 1.0)
    upper_delta = 1.0 - upper_base.clamp_min(0).pow(mutation_power)
    delta = torch.where(random <= 0.5, lower_delta, upper_delta)
    numeric_candidate = (population + delta * span).clamp(min=spec.lower, max=spec.upper)

    alternatives = torch.floor(random * (categorical_counts - 1).clamp_min(1)).to(torch.long) + 1
    current = torch.round(population - spec.lower).to(torch.long)
    categorical_candidate = (
        torch.remainder(current + alternatives, categorical_counts.clamp_min(1)).to(
            population.dtype
        )
        + spec.lower
    )
    candidate = torch.where(mutable_numeric, numeric_candidate, categorical_candidate)
    return _repair_population_unchecked(torch.where(mutate, candidate, population), spec)
