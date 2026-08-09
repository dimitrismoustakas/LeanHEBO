# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Elitist NSGA-II parent-offspring survival."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from leanhebo.search.duplicates import _duplicate_mask_canonical, duplicate_mask
from leanhebo.search.repair import MixedVariableSpec
from leanhebo.search.sorting import crowding_distance, non_dominated_sort


@dataclass(frozen=True, slots=True)
class SurvivalSelection:
    """Indices and metrics produced while truncating one candidate pool."""

    indices: Tensor
    ranks: Tensor
    crowding: Tensor


@dataclass(frozen=True, slots=True)
class SurvivalResult:
    """Selected population, objectives, and source-pool metadata."""

    population: Tensor
    objectives: Tensor
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


def survival_indices(objectives: Tensor, n_survive: int) -> Tensor:
    """Return only the candidate indices selected by NSGA-II survival."""

    return select_survivors(objectives, n_survive).indices


def elitist_survival(
    population: Tensor,
    objectives: Tensor,
    n_survive: int,
    *,
    spec: MixedVariableSpec | None = None,
    eliminate_duplicate_points: bool = True,
    duplicate_tolerance: float = 0.0,
    _population_is_canonical: bool = False,
) -> SurvivalResult:
    """Apply canonical deduplication followed by rank-and-crowding survival."""

    if population.ndim != 2:
        raise ValueError("population must have shape [population, dimensions]")
    if objectives.ndim != 2 or objectives.shape[0] != population.shape[0]:
        raise ValueError("objectives must have one row for each population member")
    if population.device != objectives.device:
        raise ValueError("population and objectives must be on the same device")
    if n_survive < 0:
        raise ValueError("n_survive must be non-negative")

    source_indices = torch.arange(population.shape[0], device=population.device)
    if eliminate_duplicate_points:
        if _population_is_canonical:
            keep = ~_duplicate_mask_canonical(
                population,
                existing=None,
                spec=spec,
                atol=duplicate_tolerance,
            )
        else:
            keep = ~duplicate_mask(
                population,
                spec=spec,
                atol=duplicate_tolerance,
            )
        source_indices = source_indices[keep]
        population = population[keep]
        objectives = objectives[keep]
    target = min(n_survive, population.shape[0])
    selection = select_survivors(objectives, target)
    selected = selection.indices
    return SurvivalResult(
        population=population[selected],
        objectives=objectives[selected],
        indices=source_indices[selected],
        ranks=selection.ranks[selected],
        crowding=selection.crowding[selected],
    )
