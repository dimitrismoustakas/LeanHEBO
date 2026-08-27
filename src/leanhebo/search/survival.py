# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Rank-and-crowding NSGA-II survival selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from leanhebo.search.sorting import crowding_distance, non_dominated_sort


@dataclass(frozen=True, slots=True)
class SurvivalSelection:
    """Indices and metrics produced while truncating one candidate pool."""

    indices: Tensor
    ranks: Tensor
    crowding: Tensor


def select_survivors(objectives: Tensor, n_survive: int) -> SurvivalSelection:
    """Select complete fronts and crowding-best members of the split front."""

    if n_survive < 0:
        raise ValueError("n_survive must be non-negative")
    if objectives.ndim != 2:
        raise ValueError("objectives must have shape [population, objectives]")
    if n_survive > objectives.shape[0]:
        raise ValueError("n_survive cannot exceed the candidate-pool size")

    ranks = non_dominated_sort(objectives)
    crowding = crowding_distance(objectives, ranks)
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
