# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Canonical duplicate detection for dense mixed-variable populations."""

from __future__ import annotations

import math
from typing import Literal, overload

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


def _pairwise_equal(
    left: Tensor,
    right: Tensor,
    *,
    spec: MixedVariableSpec | None,
    atol: float,
) -> Tensor:
    if atol == 0:
        return (left[:, None, :] == right[None, :, :]).all(dim=-1)
    difference = (left[:, None, :] - right[None, :, :]).abs()
    if spec is None:
        return (difference <= atol).all(dim=-1)

    discrete = spec.integer_mask | spec.categorical_mask | spec.fixed_mask
    equal = torch.ones(
        (left.shape[0], right.shape[0]),
        dtype=torch.bool,
        device=left.device,
    )
    if bool(discrete.any()):
        equal &= (difference[..., discrete] == 0).all(dim=-1)
    continuous = ~discrete
    if bool(continuous.any()):
        equal &= (difference[..., continuous] <= atol).all(dim=-1)
    return equal


def duplicate_mask(
    population: Tensor,
    *,
    existing: Tensor | None = None,
    spec: MixedVariableSpec | None = None,
    atol: float = 0.0,
) -> Tensor:
    """Mark repeated rows, preserving the first occurrence in ``population``.

    Rows matching ``existing`` are always marked. When a specification is provided, both sets are
    repaired before comparison, so values such as ``1.1`` and ``0.9`` are the same canonical integer
    point. ``atol`` applies only to continuous dimensions when metadata is available.
    """

    _validate_population(population, name="population")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be non-negative and finite")
    if spec is not None:
        if population.shape[1] != spec.dimension:
            raise ValueError("population dimensionality does not match the search specification")
        if population.device != spec.lower.device or population.dtype != spec.lower.dtype:
            raise ValueError("population and search specification must share device and dtype")
        canonical = repair_population(population, spec)
    else:
        canonical = population

    count = population.shape[0]
    equal_within = _pairwise_equal(canonical, canonical, spec=spec, atol=atol)
    index = torch.arange(count, device=population.device)
    earlier_equal = equal_within & (index[None, :] < index[:, None])
    if atol == 0:
        has_earlier_copy = earlier_equal.any(dim=1)
    else:
        # Approximate equality is not transitive. Compare only against rows retained so far, so a
        # chain A~=B, B~=C cannot incorrectly remove C when A and C are farther than ``atol``.
        has_earlier_copy = torch.zeros(count, dtype=torch.bool, device=population.device)
        for row in range(1, count):
            retained = ~has_earlier_copy[:row]
            has_earlier_copy[row] = (earlier_equal[row, :row] & retained).any()

    if existing is None:
        return has_earlier_copy
    _validate_population(existing, name="existing")
    if existing.shape[1:] != population.shape[1:]:
        raise ValueError("existing and population must have the same dimensionality")
    if existing.device != population.device or existing.dtype != population.dtype:
        raise ValueError("existing and population must share device and dtype")
    canonical_existing = repair_population(existing, spec) if spec is not None else existing
    matches_existing = _pairwise_equal(
        canonical,
        canonical_existing,
        spec=spec,
        atol=atol,
    ).any(dim=1)
    return has_earlier_copy | matches_existing


@overload
def eliminate_duplicates(
    population: Tensor,
    *,
    existing: Tensor | None = None,
    spec: MixedVariableSpec | None = None,
    atol: float = 0.0,
    return_indices: Literal[False] = False,
) -> Tensor: ...


@overload
def eliminate_duplicates(
    population: Tensor,
    *,
    existing: Tensor | None = None,
    spec: MixedVariableSpec | None = None,
    atol: float = 0.0,
    return_indices: Literal[True],
) -> tuple[Tensor, Tensor]: ...


def eliminate_duplicates(
    population: Tensor,
    *,
    existing: Tensor | None = None,
    spec: MixedVariableSpec | None = None,
    atol: float = 0.0,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Return unique rows in first-occurrence order, optionally with source indices."""

    keep = ~duplicate_mask(population, existing=existing, spec=spec, atol=atol)
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    unique = population[indices]
    if return_indices:
        return unique, indices
    return unique


# Readable alias for callers that only need the Boolean mask.
canonical_duplicate_mask = duplicate_mask
