# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Evolutionary operators whose mutation follows conditional activity."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leanhebo.search.operators import (
    _validate_generator,
    _validate_probability,
)
from leanhebo.search.repair import (
    MixedVariableSpec,
    _repair_population_unchecked,
    repair_population,
)

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
    """Mutate active, nonfixed genes with a default probability computed per row."""

    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite")
    if probability is not None:
        _validate_probability(probability, name="probability")
    _validate_generator(generator, population.device)

    return _conditional_mutation(
        population,
        spec,
        semantics,
        probability=probability,
        eta=eta,
        generator=generator,
        validate=True,
    )


def _conditional_mutation_unchecked(
    population: Tensor,
    spec: MixedVariableSpec,
    semantics: ConditionalSearchSemantics,
    *,
    probability: float | None,
    eta: float,
    generator: torch.Generator | None,
) -> Tensor:
    return _conditional_mutation(
        population,
        spec,
        semantics,
        probability=probability,
        eta=eta,
        generator=generator,
        validate=False,
    )


def _conditional_mutation(
    population: Tensor,
    spec: MixedVariableSpec,
    semantics: ConditionalSearchSemantics,
    *,
    probability: float | None,
    eta: float,
    generator: torch.Generator | None,
    validate: bool,
) -> Tensor:

    canonical = (
        repair_population(population, spec)
        if validate
        else _repair_population_unchecked(population, spec)
    )
    if canonical.shape[0] == 0 or probability == 0:
        return canonical
    active = (
        _validated_activity_mask(canonical, semantics)
        if validate
        else semantics.activity_mask(canonical)
    )
    categorical_counts = torch.round(spec.upper - spec.lower + 1.0).to(torch.long)
    mutable_numeric = spec.mutable_numeric_mask & (spec.upper > spec.lower)
    mutable_categorical = spec.mutable_categorical_mask & (categorical_counts > 1)
    mutable = active & (mutable_numeric | mutable_categorical)

    if probability is None:
        mutable_counts = mutable.sum(dim=1, keepdim=True)
        row_probability = torch.where(
            mutable_counts > 0,
            mutable_counts.reciprocal().to(dtype=canonical.dtype),
            torch.zeros_like(mutable_counts, dtype=canonical.dtype),
        )
    else:
        row_probability = canonical.new_full((canonical.shape[0], 1), probability)

    mutate = (
        torch.rand(canonical.shape, device=canonical.device, generator=generator) < row_probability
    ) & mutable

    span = spec.upper - spec.lower
    safe_span = span.clamp_min(torch.finfo(canonical.dtype).eps)
    delta_lower = (canonical - spec.lower) / safe_span
    delta_upper = (spec.upper - canonical) / safe_span
    random = torch.rand(
        canonical.shape,
        dtype=canonical.dtype,
        device=canonical.device,
        generator=generator,
    )
    mutation_power = 1.0 / (eta + 1.0)
    lower_base = 2.0 * random + (1.0 - 2.0 * random) * (1.0 - delta_lower).pow(eta + 1.0)
    lower_delta = lower_base.clamp_min(0).pow(mutation_power) - 1.0
    upper_base = 2.0 * (1.0 - random) + 2.0 * (random - 0.5) * (1.0 - delta_upper).pow(eta + 1.0)
    upper_delta = 1.0 - upper_base.clamp_min(0).pow(mutation_power)
    delta = torch.where(random <= 0.5, lower_delta, upper_delta)
    numeric_candidate = (canonical + delta * span).clamp(min=spec.lower, max=spec.upper)

    alternatives = torch.floor(random * (categorical_counts - 1).clamp_min(1)).to(torch.long) + 1
    current = torch.round(canonical - spec.lower).to(torch.long)
    categorical_candidate = (
        torch.remainder(current + alternatives, categorical_counts.clamp_min(1)).to(canonical.dtype)
        + spec.lower
    )
    candidate = torch.where(mutable_numeric, numeric_candidate, categorical_candidate)
    result = torch.where(mutate, candidate, canonical)
    return (
        repair_population(result, spec) if validate else _repair_population_unchecked(result, spec)
    )
