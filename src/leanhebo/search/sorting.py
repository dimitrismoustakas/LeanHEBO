# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Tensor-native dominance, non-dominated sorting, and crowding."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_objectives(objectives: Tensor) -> None:
    if objectives.ndim != 2:
        raise ValueError(
            f"objectives must have shape [population, objectives], got {tuple(objectives.shape)}"
        )
    if objectives.shape[1] == 0:
        raise ValueError("at least one objective is required")
    if not objectives.is_floating_point():
        raise TypeError("objectives must use a floating-point dtype")
    if not bool(torch.isfinite(objectives).all()):
        raise ValueError("objectives contain non-finite values")


def dominance_matrix(objectives: Tensor) -> Tensor:
    """Return ``matrix[i, j] == True`` when point ``i`` dominates point ``j``.

    All objectives are minimized. Equality alone is not dominance.
    """

    _validate_objectives(objectives)
    no_worse = (objectives[:, None, :] <= objectives[None, :, :]).all(dim=-1)
    strictly_better = (objectives[:, None, :] < objectives[None, :, :]).any(dim=-1)
    return no_worse & strictly_better


def _sort_from_dominance(dominates: Tensor) -> tuple[Tensor, list[Tensor]]:
    population_size = dominates.shape[0]
    ranks = torch.full(
        (population_size,),
        -1,
        dtype=torch.long,
        device=dominates.device,
    )
    if population_size == 0:
        return ranks, []

    domination_count = dominates.sum(dim=0, dtype=torch.long)
    current = torch.nonzero(domination_count == 0, as_tuple=False).flatten()
    fronts: list[Tensor] = []
    rank = 0

    while current.numel():
        ranks[current] = rank
        fronts.append(current)
        domination_count = domination_count - dominates[current].sum(dim=0, dtype=torch.long)
        current = torch.nonzero((domination_count == 0) & (ranks < 0), as_tuple=False).flatten()
        rank += 1

    if bool((ranks < 0).any()):  # Defensive: strict dominance should always form a DAG.
        raise RuntimeError("non-dominated sorting failed to assign every point")
    return ranks, fronts


def non_dominated_sort(objectives: Tensor) -> Tensor:
    """Assign zero-based Pareto ranks to a minimization objective matrix."""

    ranks, _ = _sort_from_dominance(dominance_matrix(objectives))
    return ranks


def non_dominated_fronts(objectives: Tensor) -> list[Tensor]:
    """Return index tensors for Pareto fronts in increasing rank order."""

    _, fronts = _sort_from_dominance(dominance_matrix(objectives))
    return fronts


# Explicit naming used by some integration layers.
non_dominated_ranks = non_dominated_sort
fast_non_dominated_sort = non_dominated_sort


def _front_crowding(objectives: Tensor) -> Tensor:
    size, objective_count = objectives.shape
    distance = torch.zeros(size, dtype=objectives.dtype, device=objectives.device)
    if size == 0:
        return distance
    if size <= 2:
        return torch.full_like(distance, torch.inf)

    for objective_index in range(objective_count):
        values = objectives[:, objective_index]
        order = torch.argsort(values, stable=True)
        span = values[order[-1]] - values[order[0]]
        if bool(span <= 0):
            continue
        distance[order[0]] = torch.inf
        distance[order[-1]] = torch.inf
        interior = (values[order[2:]] - values[order[:-2]]) / span
        distance[order[1:-1]] += interior
    return distance


def _ranked_crowding(objectives: Tensor, ranks: Tensor) -> Tensor:
    """Compute crowding for every front without a Python loop over fronts.

    A stable value sort followed by a stable rank sort produces the same per-front ordering as
    sorting each front independently.  Keeping fronts in one tensor avoids hundreds of tiny sort,
    indexing, and allocation calls when an NSGA-II population contains many Pareto fronts.
    """

    population_size, objective_count = objectives.shape
    distance = torch.zeros(
        population_size,
        dtype=objectives.dtype,
        device=objectives.device,
    )
    if population_size == 0:
        return distance

    _, inverse, counts = torch.unique(
        ranks,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    distance[counts[inverse] <= 2] = torch.inf

    for objective_index in range(objective_count):
        values = objectives[:, objective_index]
        by_value = torch.argsort(values, stable=True)
        order = by_value[torch.argsort(ranks[by_value], stable=True)]
        ordered_ranks = ranks[order]
        ordered_values = values[order]

        first = torch.ones(population_size, dtype=torch.bool, device=objectives.device)
        first[1:] = ordered_ranks[1:] != ordered_ranks[:-1]
        last = torch.ones_like(first)
        last[:-1] = first[1:]
        group = first.cumsum(dim=0) - 1

        first_positions = torch.nonzero(first, as_tuple=False).flatten()
        last_positions = torch.nonzero(last, as_tuple=False).flatten()
        spans = ordered_values[last_positions] - ordered_values[first_positions]
        contributes = (counts > 2) & (spans > 0)

        boundary = (first | last) & contributes[group]
        distance[order[boundary]] = torch.inf

        interior = ~(first | last) & contributes[group]
        positions = torch.nonzero(interior, as_tuple=False).flatten()
        if positions.numel():
            increments = (ordered_values[positions + 1] - ordered_values[positions - 1]) / spans[
                group[positions]
            ]
            distance[order[positions]] += increments
    return distance


def crowding_distance(objectives: Tensor, ranks: Tensor | None = None) -> Tensor:
    """Compute NSGA-II crowding distances within each Pareto front.

    If ``ranks`` is omitted, the entire objective matrix is treated as one front. Objective ranges
    of zero add no distance, avoiding arbitrary boundary preference for a constant objective.
    """

    _validate_objectives(objectives)
    population_size = objectives.shape[0]
    if ranks is None:
        return _front_crowding(objectives)
    if ranks.shape != (population_size,) or ranks.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("ranks must be a one-dimensional integer tensor matching the population")
    if ranks.device != objectives.device:
        raise ValueError("ranks and objectives must be on the same device")
    if bool((ranks < 0).any()):
        raise ValueError("ranks must be non-negative")

    return _ranked_crowding(objectives, ranks)
