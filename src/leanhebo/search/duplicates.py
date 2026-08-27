# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Canonical duplicate detection for dense mixed-variable populations."""

from __future__ import annotations

import torch
from torch import Tensor

from leanhebo.search.repair import MixedVariableSpec, repair_population


def _validate_population(population: Tensor, *, name: str) -> None:
    if population.ndim != 2:
        raise ValueError(f"{name} must have shape [population, dimensions]")
    if not population.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(population).all()):
        raise ValueError(f"{name} contains non-finite values")


def _exact_duplicate_mask(population: Tensor, existing: Tensor | None = None) -> Tensor:
    """Mark exact row duplicates without materializing a quadratic comparison tensor."""

    count, dimension = population.shape
    if count == 0:
        return torch.zeros(0, dtype=torch.bool, device=population.device)
    if dimension == 0:
        # ``torch.unique(..., dim=0)`` rejects zero-column tensors. Exact row comparison treats
        # every such row as equal, so preserve the established public behavior explicitly.
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


def duplicate_mask(
    population: Tensor,
    *,
    existing: Tensor | None = None,
    spec: MixedVariableSpec | None = None,
) -> Tensor:
    """Mark exactly repeated rows, preserving the first occurrence in ``population``.

    Rows matching ``existing`` are always marked. When a specification is provided, both sets are
    repaired before comparison, so values such as ``1.1`` and ``0.9`` are the same canonical integer
    point.
    """

    _validate_population(population, name="population")
    if spec is not None:
        if population.shape[1] != spec.dimension:
            raise ValueError("population dimensionality does not match the search specification")
        if population.device != spec.lower.device or population.dtype != spec.lower.dtype:
            raise ValueError("population and search specification must share device and dtype")
        canonical = repair_population(population, spec)
    else:
        canonical = population

    canonical_existing: Tensor | None = None
    if existing is not None:
        _validate_population(existing, name="existing")
        if existing.shape[1:] != population.shape[1:]:
            raise ValueError("existing and population must have the same dimensionality")
        if existing.device != population.device or existing.dtype != population.dtype:
            raise ValueError("existing and population must share device and dtype")
        canonical_existing = repair_population(existing, spec) if spec is not None else existing
    return _exact_duplicate_mask(canonical, canonical_existing)
