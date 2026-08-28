# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Selection, crossover, and mutation operators for dense Torch populations.

These operators trust their tensor contracts: canonical populations, matching devices
and dtypes, and probabilities validated by the search entry points (`TorchNSGA2`).
They sit below the public search boundary and do not re-validate per call.
"""

from __future__ import annotations

import torch
from torch import Tensor

from leanhebo.search.repair import MixedVariableSpec, _repair_population_unchecked


def _validate_generator(generator: torch.Generator | None, device: torch.device) -> None:
    if generator is None:
        return
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError(
            f"generator is on {generator_device.type}, but search tensors are on {device.type}"
        )
    if device.type == "cuda" and generator_device.index not in (None, device.index):
        raise ValueError(f"generator is on {generator_device}, but search tensors are on {device}")


def binary_tournament(
    ranks: Tensor,
    crowding: Tensor,
    n_select: int,
    *,
    generator: torch.Generator | None = None,
    tournament_size: int = 2,
) -> Tensor:
    """Select indices by Pareto rank, then crowding distance, with random tie breaks."""

    contestants = torch.randint(
        ranks.numel(),
        (n_select, tournament_size),
        device=ranks.device,
        generator=generator,
    )
    winner = contestants[:, 0]
    for column in range(1, tournament_size):
        challenger = contestants[:, column]
        winner_rank = ranks[winner]
        challenger_rank = ranks[challenger]
        winner_crowding = crowding[winner]
        challenger_crowding = crowding[challenger]

        better = (challenger_rank < winner_rank) | (
            (challenger_rank == winner_rank) & (challenger_crowding > winner_crowding)
        )
        tied = (challenger_rank == winner_rank) & (challenger_crowding == winner_crowding)
        random_tie_break = (
            torch.rand(
                n_select,
                device=ranks.device,
                generator=generator,
            )
            < 0.5
        )
        winner = torch.where(better | (tied & random_tie_break), challenger, winner)
    return winner


def sbx_crossover(
    parent_a: Tensor,
    parent_b: Tensor,
    lower: Tensor,
    upper: Tensor,
    *,
    numeric_mask: Tensor | None = None,
    probability: float = 0.9,
    dimension_probability: float = 0.5,
    eta: float = 15.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply bounded simulated-binary crossover to selected numeric columns.

    One crossover gate is sampled per parent pair and a second per dimension. Children are always
    clipped to the provided bounds. Integer rounding is intentionally left to mixed-variable repair.
    """

    if numeric_mask is None:
        numeric_mask = torch.ones_like(lower, dtype=torch.bool)

    pair_count, dimension = parent_a.shape
    if pair_count == 0:
        return parent_a.clone(), parent_b.clone()

    x_low = torch.minimum(parent_a, parent_b)
    x_high = torch.maximum(parent_a, parent_b)
    difference = x_high - x_low
    epsilon = torch.finfo(parent_a.dtype).eps
    safe_difference = difference.clamp_min(epsilon)
    mutable = numeric_mask & (upper > lower)
    pair_gate = (
        torch.rand(
            (pair_count, 1),
            device=parent_a.device,
            generator=generator,
        )
        < probability
    )
    dimension_gate = (
        torch.rand(
            (pair_count, dimension),
            device=parent_a.device,
            generator=generator,
        )
        < dimension_probability
    )
    crossed = pair_gate & dimension_gate & mutable & (difference > epsilon)

    random = torch.rand(
        (pair_count, dimension),
        dtype=parent_a.dtype,
        device=parent_a.device,
        generator=generator,
    )
    exponent = 1.0 / (eta + 1.0)

    beta_left = 1.0 + 2.0 * (x_low - lower) / safe_difference
    alpha_left = 2.0 - beta_left.pow(-(eta + 1.0))
    beta_q_left = torch.where(
        random <= 1.0 / alpha_left,
        (random * alpha_left).clamp_min(0).pow(exponent),
        (1.0 / (2.0 - random * alpha_left)).pow(exponent),
    )
    candidate_a = 0.5 * (x_low + x_high - beta_q_left * difference)

    beta_right = 1.0 + 2.0 * (upper - x_high) / safe_difference
    alpha_right = 2.0 - beta_right.pow(-(eta + 1.0))
    beta_q_right = torch.where(
        random <= 1.0 / alpha_right,
        (random * alpha_right).clamp_min(0).pow(exponent),
        (1.0 / (2.0 - random * alpha_right)).pow(exponent),
    )
    candidate_b = 0.5 * (x_low + x_high + beta_q_right * difference)
    candidate_a = candidate_a.clamp(min=lower, max=upper)
    candidate_b = candidate_b.clamp(min=lower, max=upper)

    swap = (
        torch.rand(
            (pair_count, dimension),
            device=parent_a.device,
            generator=generator,
        )
        < 0.5
    )
    first_candidate = torch.where(swap, candidate_b, candidate_a)
    second_candidate = torch.where(swap, candidate_a, candidate_b)
    child_a = torch.where(crossed, first_candidate, parent_a)
    child_b = torch.where(crossed, second_candidate, parent_b)
    return child_a, child_b


def uniform_categorical_crossover(
    parent_a: Tensor,
    parent_b: Tensor,
    categorical_mask: Tensor,
    *,
    probability: float = 0.9,
    dimension_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Uniformly exchange selected categorical genes between parent pairs."""

    if parent_a.shape[0] == 0 or not bool(categorical_mask.any()):
        return parent_a.clone(), parent_b.clone()

    pair_gate = (
        torch.rand(
            (parent_a.shape[0], 1),
            device=parent_a.device,
            generator=generator,
        )
        < probability
    )
    swap = (
        torch.rand(
            parent_a.shape,
            device=parent_a.device,
            generator=generator,
        )
        < dimension_probability
    )
    swap &= pair_gate & categorical_mask
    return torch.where(swap, parent_b, parent_a), torch.where(swap, parent_a, parent_b)


def mixed_variable_crossover(
    parent_a: Tensor,
    parent_b: Tensor,
    spec: MixedVariableSpec,
    *,
    probability: float = 0.9,
    dimension_probability: float = 0.5,
    eta: float = 15.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Combine SBX on numeric columns with uniform categorical crossover."""

    child_a, child_b = sbx_crossover(
        parent_a,
        parent_b,
        spec.lower,
        spec.upper,
        numeric_mask=spec.mutable_numeric_mask,
        probability=probability,
        dimension_probability=dimension_probability,
        eta=eta,
        generator=generator,
    )
    if spec.has_categorical:
        categorical = spec.mutable_categorical_mask
        categorical_a, categorical_b = uniform_categorical_crossover(
            parent_a,
            parent_b,
            categorical,
            probability=probability,
            dimension_probability=dimension_probability,
            generator=generator,
        )
        child_a = torch.where(categorical, categorical_a, child_a)
        child_b = torch.where(categorical, categorical_b, child_b)
    return (
        _repair_population_unchecked(child_a, spec),
        _repair_population_unchecked(child_b, spec),
    )


def polynomial_mutation(
    population: Tensor,
    lower: Tensor,
    upper: Tensor,
    *,
    numeric_mask: Tensor | None = None,
    probability: float | None = None,
    eta: float = 20.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply bounded polynomial mutation to numeric dimensions."""

    if numeric_mask is None:
        numeric_mask = torch.ones_like(lower, dtype=torch.bool)
    mutable = numeric_mask & (upper > lower)
    if probability is None:
        mutable_count = int(mutable.sum().item())
        probability = 0.0 if mutable_count == 0 else 1.0 / mutable_count
    if population.shape[0] == 0 or probability == 0:
        return population.clone()

    span = upper - lower
    safe_span = span.clamp_min(torch.finfo(population.dtype).eps)
    clipped = population.clamp(min=lower, max=upper)
    delta_lower = (clipped - lower) / safe_span
    delta_upper = (upper - clipped) / safe_span
    random = torch.rand(
        population.shape,
        dtype=population.dtype,
        device=population.device,
        generator=generator,
    )
    mutate = (
        torch.rand(population.shape, device=population.device, generator=generator) < probability
    ) & mutable
    mutation_power = 1.0 / (eta + 1.0)

    lower_branch = random <= 0.5
    lower_base = 2.0 * random + (1.0 - 2.0 * random) * (1.0 - delta_lower).pow(eta + 1.0)
    lower_delta = lower_base.clamp_min(0).pow(mutation_power) - 1.0
    upper_base = 2.0 * (1.0 - random) + 2.0 * (random - 0.5) * (1.0 - delta_upper).pow(eta + 1.0)
    upper_delta = 1.0 - upper_base.clamp_min(0).pow(mutation_power)
    delta = torch.where(lower_branch, lower_delta, upper_delta)
    candidate = (clipped + delta * span).clamp(min=lower, max=upper)
    return torch.where(mutate, candidate, population)


def categorical_mutation(
    population: Tensor,
    lower: Tensor,
    upper: Tensor,
    categorical_mask: Tensor,
    *,
    probability: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Replace selected categorical codes with a uniformly sampled alternative code."""

    counts = torch.round(upper - lower + 1.0).to(torch.long)
    mutable = categorical_mask & (counts > 1)
    if probability is None:
        mutable_count = int(mutable.sum().item())
        probability = 0.0 if mutable_count == 0 else 1.0 / mutable_count
    if population.shape[0] == 0 or probability == 0:
        return population.clone()

    mutate = (
        torch.rand(population.shape, device=population.device, generator=generator) < probability
    ) & mutable
    random = torch.rand(
        population.shape,
        dtype=population.dtype,
        device=population.device,
        generator=generator,
    )
    alternatives = torch.floor(random * (counts - 1).clamp_min(1)).to(torch.long) + 1
    current = torch.round(population - lower).to(torch.long)
    replacement = torch.remainder(current + alternatives, counts.clamp_min(1))
    replacement = replacement.to(population.dtype) + lower
    return torch.where(mutate, replacement, population)


def mutate_population(
    population: Tensor,
    spec: MixedVariableSpec,
    *,
    probability: float | None = None,
    eta: float = 20.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply numeric and alternative-category mutation, then canonical repair."""

    if probability is None:
        probability = 0.0 if spec.mutable_count == 0 else 1.0 / spec.mutable_count
    if spec.has_numeric:
        population = polynomial_mutation(
            population,
            spec.lower,
            spec.upper,
            numeric_mask=spec.mutable_numeric_mask,
            probability=probability,
            eta=eta,
            generator=generator,
        )
    if spec.has_categorical:
        population = categorical_mutation(
            population,
            spec.lower,
            spec.upper,
            spec.mutable_categorical_mask,
            probability=probability,
            generator=generator,
        )
    return _repair_population_unchecked(population, spec)
