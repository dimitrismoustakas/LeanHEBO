# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Selection, crossover, and mutation operators for dense Torch populations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from leanhebo.search.repair import MixedVariableSpec, repair_population


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


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

    if ranks.ndim != 1 or crowding.shape != ranks.shape:
        raise ValueError("ranks and crowding must be matching one-dimensional tensors")
    if ranks.device != crowding.device:
        raise ValueError("ranks and crowding must be on the same device")
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if ranks.dtype not in integer_dtypes or not crowding.is_floating_point():
        raise TypeError("ranks must be integer and crowding must be floating-point")
    if bool((ranks < 0).any()):
        raise ValueError("ranks must be non-negative")
    if bool(torch.isnan(crowding).any()):
        raise ValueError("crowding must not contain NaN")
    if isinstance(n_select, bool) or not isinstance(n_select, int) or n_select < 0:
        raise ValueError("n_select must be non-negative")
    if (
        isinstance(tournament_size, bool)
        or not isinstance(tournament_size, int)
        or tournament_size < 2
    ):
        raise ValueError("tournament_size must be at least two")
    if n_select == 0:
        return torch.empty(0, dtype=torch.long, device=ranks.device)
    if ranks.numel() == 0:
        raise ValueError("cannot select from an empty population")

    _validate_generator(generator, ranks.device)
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


def _validate_parent_tensors(
    parent_a: Tensor,
    parent_b: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> None:
    if parent_a.ndim != 2 or parent_b.shape != parent_a.shape:
        raise ValueError("parent tensors must have the same [pairs, dimensions] shape")
    if lower.shape != (parent_a.shape[1],) or upper.shape != lower.shape:
        raise ValueError("bounds must contain one value per parent dimension")
    if not parent_a.is_floating_point() or not parent_b.is_floating_point():
        raise TypeError("parent tensors must be floating-point")
    tensors = (parent_b, lower, upper)
    if any(tensor.device != parent_a.device for tensor in tensors):
        raise ValueError("parents and bounds must share a device")
    if any(tensor.dtype != parent_a.dtype for tensor in tensors):
        raise ValueError("parents and bounds must share a dtype")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (parent_a, parent_b, lower, upper)):
        raise ValueError("parents and bounds must contain only finite values")
    if bool((lower > upper).any()):
        raise ValueError("lower bounds must not exceed upper bounds")


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

    _validate_parent_tensors(parent_a, parent_b, lower, upper)
    _validate_probability(probability, name="probability")
    _validate_probability(dimension_probability, name="dimension_probability")
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite")
    _validate_generator(generator, parent_a.device)

    if numeric_mask is None:
        numeric_mask = torch.ones_like(lower, dtype=torch.bool)
    else:
        numeric_mask = torch.as_tensor(numeric_mask, dtype=torch.bool, device=parent_a.device)
        if numeric_mask.shape != lower.shape:
            raise ValueError("numeric_mask must contain one entry per dimension")

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


# Descriptive alias used in documentation.
simulated_binary_crossover = sbx_crossover


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

    if parent_a.ndim != 2 or parent_b.shape != parent_a.shape:
        raise ValueError("parent tensors must have the same [pairs, dimensions] shape")
    if parent_a.device != parent_b.device or parent_a.dtype != parent_b.dtype:
        raise ValueError("parent tensors must share device and dtype")
    if not parent_a.is_floating_point():
        raise TypeError("parent tensors must be floating-point")
    categorical_mask = torch.as_tensor(
        categorical_mask,
        dtype=torch.bool,
        device=parent_a.device,
    )
    if categorical_mask.shape != (parent_a.shape[1],):
        raise ValueError("categorical_mask must contain one entry per dimension")
    _validate_probability(probability, name="probability")
    _validate_probability(dimension_probability, name="dimension_probability")
    _validate_generator(generator, parent_a.device)
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
    return repair_population(child_a, spec), repair_population(child_b, spec)


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

    _validate_parent_tensors(population, population, lower, upper)
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite")
    _validate_generator(generator, population.device)
    if numeric_mask is None:
        numeric_mask = torch.ones_like(lower, dtype=torch.bool)
    else:
        numeric_mask = torch.as_tensor(numeric_mask, dtype=torch.bool, device=population.device)
        if numeric_mask.shape != lower.shape:
            raise ValueError("numeric_mask must contain one entry per dimension")
    mutable = numeric_mask & (upper > lower)
    mutable_count = int(mutable.sum().item())
    if probability is None:
        probability = 0.0 if mutable_count == 0 else 1.0 / mutable_count
    _validate_probability(probability, name="probability")
    if population.shape[0] == 0 or mutable_count == 0 or probability == 0:
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

    _validate_parent_tensors(population, population, lower, upper)
    categorical_mask = torch.as_tensor(
        categorical_mask,
        dtype=torch.bool,
        device=population.device,
    )
    if categorical_mask.shape != lower.shape:
        raise ValueError("categorical_mask must contain one entry per dimension")
    _validate_generator(generator, population.device)

    counts = torch.round(upper - lower + 1.0).to(torch.long)
    mutable = categorical_mask & (counts > 1)
    mutable_count = int(mutable.sum().item())
    if probability is None:
        probability = 0.0 if mutable_count == 0 else 1.0 / mutable_count
    _validate_probability(probability, name="probability")
    if population.shape[0] == 0 or mutable_count == 0 or probability == 0:
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
        mutable_count = int((~spec.fixed_mask).sum().item())
        probability = 0.0 if mutable_count == 0 else 1.0 / mutable_count
    numeric = polynomial_mutation(
        population,
        spec.lower,
        spec.upper,
        numeric_mask=spec.mutable_numeric_mask,
        probability=probability,
        eta=eta,
        generator=generator,
    )
    mixed = categorical_mutation(
        numeric,
        spec.lower,
        spec.upper,
        spec.mutable_categorical_mask,
        probability=probability,
        generator=generator,
    )
    return repair_population(mixed, spec)


# Generic short spelling for external integration.
numeric_mutation = polynomial_mutation
