# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Tensor-native dominance, non-dominated sorting, and crowding."""

from __future__ import annotations

import torch
from torch import Tensor


def _dominance_matrix_unchecked(objectives: Tensor) -> Tensor:
    """Return the strict Pareto dominance relation for finite minimization objectives."""

    no_worse = (objectives[:, None, :] <= objectives[None, :, :]).all(dim=-1)
    # For finite values, i is strictly better than j in at least one objective exactly when
    # i is no worse than j but j is not no worse than i. Reusing the transpose avoids a second
    # population-by-population-by-objective comparison without changing tie behavior.
    return no_worse & ~no_worse.mT


def _non_dominated_sort_unchecked(
    objectives: Tensor,
    stop_after: int | None = None,
) -> tuple[Tensor, int | None]:
    """Rank an internal objective matrix, optionally replacing unused tail ranks by a sentinel."""

    dominates = _dominance_matrix_unchecked(objectives)
    population_size = dominates.shape[0]
    ranks = torch.full(
        (population_size,),
        -1,
        dtype=torch.long,
        device=dominates.device,
    )
    domination_count = dominates.sum(dim=0, dtype=torch.long)
    current = torch.nonzero(domination_count == 0, as_tuple=False).flatten()
    rank = 0
    ranked_count = 0

    while current.numel():
        ranks[current] = rank
        ranked_count += current.numel()
        if stop_after is not None and ranked_count >= stop_after and ranked_count < population_size:
            tail_rank = rank + 1
            ranks[ranks < 0] = tail_rank
            return ranks, tail_rank
        domination_count = domination_count - dominates[current].sum(dim=0, dtype=torch.long)
        current = torch.nonzero((domination_count == 0) & (ranks < 0), as_tuple=False).flatten()
        rank += 1

    if bool((ranks < 0).any()):  # Defensive: strict dominance should always form a DAG.
        raise RuntimeError("non-dominated sorting failed to assign every point")
    return ranks, None


def _crowding_unchecked(objectives: Tensor, ranks: Tensor | None = None) -> Tensor:
    """Sum normalized neighbor gaps within each front, sorting all objectives together."""

    size = objectives.shape[0]
    if size <= 2:
        return objectives.new_full((size,), torch.inf)

    order = torch.argsort(objectives, dim=0, stable=True)
    first: int | Tensor = 0
    last: int | Tensor = -1
    if ranks is not None:
        _, groups, counts = torch.unique(
            ranks, sorted=True, return_inverse=True, return_counts=True
        )
        by_group = torch.argsort(groups[order], dim=0, stable=True)
        order = order.gather(0, by_group)
        last = counts.cumsum(0) - 1
        first = last - counts + 1

    values = objectives.gather(0, order)
    spans = values[last] - values[first]
    boundary = torch.zeros_like(spans).masked_fill_(spans > 0, torch.inf)
    scales = torch.where(spans > 0, spans, 1.0)
    if ranks is not None:
        # Every sorted column has the same front boundaries.
        scales = scales[groups[order[:, 0]]]
    gaps = torch.zeros_like(values)
    gaps[1:-1] = values[2:] - values[:-2]
    gaps = gaps / scales
    gaps[first] = boundary
    gaps[last] = boundary
    contributions = torch.empty_like(gaps).scatter_(0, order, gaps)
    # Parallel reductions can round tied distances differently and change tournament winners.
    first_column, *remaining_columns = contributions.unbind(dim=1)
    distance = first_column.clone()
    for column in remaining_columns:
        distance.add_(column)
    if ranks is not None:
        distance.masked_fill_(counts[groups] <= 2, torch.inf)
    return distance
