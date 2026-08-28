# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Rank-and-crowding NSGA-II survival selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from leanhebo.search.sorting import (
    _non_dominated_sort_unchecked,
    _ranked_crowding,
    _validate_objectives,
)


@dataclass(frozen=True, slots=True)
class SurvivalSelection:
    """Indices and metrics produced while truncating one candidate pool."""

    indices: Tensor
    ranks: Tensor
    crowding: Tensor


def select_survivors(objectives: Tensor, n_survive: int) -> SurvivalSelection:
    """Select complete fronts and crowding-best members of the split front.

    Ranking stops once the survivors are covered: rows past the cut share one sentinel
    tail rank and zero crowding, so per-row metrics are only meaningful for survivors.
    """

    if n_survive < 0:
        raise ValueError("n_survive must be non-negative")
    if objectives.ndim != 2:
        raise ValueError("objectives must have shape [population, objectives]")
    if n_survive > objectives.shape[0]:
        raise ValueError("n_survive cannot exceed the candidate-pool size")

    _validate_objectives(objectives)
    return _select_survivors_unchecked(objectives, n_survive)


def _select_survivors_unchecked(objectives: Tensor, n_survive: int) -> SurvivalSelection:
    """Select from an internally validated pool without ranking or crowding the unused tail."""

    ranks, tail_rank = _non_dominated_sort_unchecked(objectives, stop_after=n_survive)
    crowding = torch.zeros(
        objectives.shape[0],
        dtype=objectives.dtype,
        device=objectives.device,
    )
    covered = torch.ones_like(ranks, dtype=torch.bool) if tail_rank is None else ranks != tail_rank
    if n_survive:
        crowding[covered] = _ranked_crowding(objectives[covered], ranks[covered])
    selected: list[Tensor] = []
    selected_count = 0
    for rank in torch.unique(ranks, sorted=True):
        front = torch.nonzero(ranks == rank, as_tuple=False).flatten()
        remaining = n_survive - selected_count
        if remaining <= 0:
            break
        if front.numel() <= remaining:
            selected.append(front)
            selected_count += front.numel()
            continue
        order = torch.argsort(crowding[front], descending=True, stable=True)
        selected.append(front[order[:remaining]])
        selected_count += remaining
        break

    if selected:
        indices = torch.cat(selected)
    else:
        indices = torch.empty(0, dtype=torch.long, device=objectives.device)
    return SurvivalSelection(indices=indices, ranks=ranks, crowding=crowding)
