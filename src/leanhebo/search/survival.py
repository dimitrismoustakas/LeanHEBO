# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Rank-and-crowding NSGA-II survival selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from leanhebo.search.sorting import _crowding_unchecked, _non_dominated_sort_unchecked


@dataclass(frozen=True, slots=True)
class SurvivalSelection:
    """Indices and metrics produced while truncating one candidate pool."""

    indices: Tensor
    ranks: Tensor
    crowding: Tensor


def _select_survivors_unchecked(objectives: Tensor, n_survive: int) -> SurvivalSelection:
    """Select complete fronts and fill the remaining places by crowding distance."""

    ranks, tail_rank = _non_dominated_sort_unchecked(objectives, stop_after=n_survive)
    crowding = torch.zeros(
        objectives.shape[0],
        dtype=objectives.dtype,
        device=objectives.device,
    )
    covered = torch.ones_like(ranks, dtype=torch.bool) if tail_rank is None else ranks != tail_rank
    crowding[covered] = _crowding_unchecked(objectives[covered], ranks[covered])
    selected: list[Tensor] = []
    remaining = n_survive
    for rank in torch.unique(ranks, sorted=True):
        front = torch.nonzero(ranks == rank, as_tuple=False).flatten()
        if front.numel() > remaining:
            order = torch.argsort(crowding[front], descending=True, stable=True)
            front = front[order[:remaining]]
        selected.append(front)
        remaining -= front.numel()
        if remaining == 0:
            break
    return SurvivalSelection(indices=torch.cat(selected), ranks=ranks, crowding=crowding)
