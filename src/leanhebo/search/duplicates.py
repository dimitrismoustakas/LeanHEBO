# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Canonical duplicate detection for dense mixed-variable populations."""

from __future__ import annotations

import torch
from torch import Tensor


def _exact_duplicate_mask(population: Tensor, existing: Tensor | None = None) -> Tensor:
    """Mark exact row duplicates, giving priority to existing rows and first occurrences."""

    count, dimension = population.shape
    if count == 0:
        return torch.zeros(0, dtype=torch.bool, device=population.device)
    if dimension == 0:
        # All zero-column rows are equal, and torch.unique cannot compare them along dim 0.
        mask = torch.arange(count, device=population.device) > 0
        if existing is not None and existing.shape[0] > 0:
            mask.fill_(True)
        return mask

    if population.device.type == "cpu":
        seen: set[tuple[float, ...]] = set()
        if existing is not None:
            seen.update(tuple(row) for row in existing.tolist())
        duplicates: list[bool] = []
        for row in population.tolist():
            key = tuple(row)
            duplicates.append(key in seen)
            seen.add(key)
        return torch.tensor(duplicates, dtype=torch.bool, device=population.device)

    if existing is None or existing.shape[0] == 0:
        combined = population
        offset = 0
    else:
        combined = torch.cat((existing, population), dim=0)
        offset = existing.shape[0]

    unique, inverse = torch.unique(combined, dim=0, return_inverse=True)
    indices = torch.arange(combined.shape[0], device=population.device)
    first_indices = torch.full(
        (unique.shape[0],),
        combined.shape[0],
        dtype=torch.long,
        device=population.device,
    )
    first_indices.scatter_reduce_(
        0,
        inverse,
        indices,
        reduce="amin",
        include_self=True,
    )
    population_indices = indices[offset:]
    return first_indices[inverse[offset:]] != population_indices
