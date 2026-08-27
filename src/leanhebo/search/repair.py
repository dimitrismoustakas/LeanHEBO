# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Mixed-variable search-space metadata and tensor repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor


def _as_1d_tensor(
    value: Tensor | float | int | list[float] | tuple[float, ...],
    *,
    name: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(tensor.shape)}")
    return tensor


def _as_mask(value: Tensor | None, *, like: Tensor, name: str) -> Tensor:
    if value is None:
        return torch.zeros(like.shape, dtype=torch.bool, device=like.device)
    mask = torch.as_tensor(value, dtype=torch.bool, device=like.device)
    if mask.shape != like.shape:
        raise ValueError(f"{name} must have shape {tuple(like.shape)}, got {tuple(mask.shape)}")
    return mask


@dataclass(frozen=True, slots=True, init=False)
class MixedVariableSpec:
    """Dense metadata needed by mixed-variable evolutionary operators.

    All dimensions are represented in one floating-point population tensor. Integer and
    categorical columns are made canonical by :func:`repair_population`. ``steps`` are measured
    in optimization coordinates and anchored at ``lower``. A regular integer dimension therefore
    has step one, while continuous and categorical dimensions have step zero.

    Equal lower and upper bounds are always treated as fixed. Explicit ``fixed_mask`` entries use
    ``fixed_values`` (or ``lower`` when values are omitted), which supports contextual optimization
    without changing population shapes.
    """

    lower: Tensor
    upper: Tensor
    integer_mask: Tensor
    categorical_mask: Tensor
    steps: Tensor
    fixed_mask: Tensor
    fixed_values: Tensor
    _numeric_mask: Tensor = field(init=False, repr=False, compare=False)
    _mutable_numeric_mask: Tensor = field(init=False, repr=False, compare=False)
    _mutable_integer_mask: Tensor = field(init=False, repr=False, compare=False)
    _mutable_categorical_mask: Tensor = field(init=False, repr=False, compare=False)
    has_numeric: bool = field(init=False, repr=False, compare=False)
    has_integer: bool = field(init=False, repr=False, compare=False)
    has_categorical: bool = field(init=False, repr=False, compare=False)
    has_fixed: bool = field(init=False, repr=False, compare=False)
    mutable_count: int = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        lower: Tensor,
        upper: Tensor,
        integer_mask: Tensor | None = None,
        categorical_mask: Tensor | None = None,
        steps: Tensor | None = None,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
    ) -> None:
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "integer_mask", integer_mask)
        object.__setattr__(self, "categorical_mask", categorical_mask)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "fixed_mask", fixed_mask)
        object.__setattr__(self, "fixed_values", fixed_values)
        self.__post_init__()

    def __post_init__(self) -> None:
        steps_input = cast(Tensor | None, self.steps)
        fixed_values_input = cast(Tensor | None, self.fixed_values)
        lower = _as_1d_tensor(self.lower, name="lower")
        if not lower.is_floating_point():
            lower = lower.to(torch.get_default_dtype())
        upper = _as_1d_tensor(
            self.upper,
            name="upper",
            device=lower.device,
            dtype=lower.dtype,
        )
        if upper.shape != lower.shape:
            raise ValueError(
                f"upper must have shape {tuple(lower.shape)}, got {tuple(upper.shape)}"
            )
        if lower.numel() == 0:
            raise ValueError("a search space must contain at least one dimension")
        if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
            raise ValueError("search bounds must be finite")
        if bool((lower > upper).any()):
            raise ValueError("every lower bound must be less than or equal to its upper bound")

        integer_mask = _as_mask(self.integer_mask, like=lower, name="integer_mask")
        categorical_mask = _as_mask(
            self.categorical_mask,
            like=lower,
            name="categorical_mask",
        )
        if bool((integer_mask & categorical_mask).any()):
            raise ValueError("integer_mask and categorical_mask must not overlap")

        if steps_input is None:
            steps = integer_mask.to(dtype=lower.dtype)
        else:
            steps = _as_1d_tensor(
                steps_input,
                name="steps",
                device=lower.device,
                dtype=lower.dtype,
            )
            if steps.shape != lower.shape:
                raise ValueError(
                    f"steps must have shape {tuple(lower.shape)}, got {tuple(steps.shape)}"
                )
        if not bool(torch.isfinite(steps).all()) or bool((steps < 0).any()):
            raise ValueError("steps must be finite and non-negative")
        if bool((integer_mask & (steps <= 0)).any()):
            raise ValueError("integer dimensions require a positive step")

        explicit_fixed = _as_mask(self.fixed_mask, like=lower, name="fixed_mask")
        fixed_mask = explicit_fixed | (lower == upper)
        if fixed_values_input is None:
            fixed_values = lower.clone()
        else:
            fixed_values = _as_1d_tensor(
                fixed_values_input,
                name="fixed_values",
                device=lower.device,
                dtype=lower.dtype,
            )
            if fixed_values.shape != lower.shape:
                raise ValueError(
                    "fixed_values must contain one value for every population dimension"
                )
        if bool((fixed_mask & ~torch.isfinite(fixed_values)).any()):
            raise ValueError("fixed values must be finite")
        if bool((fixed_mask & ((fixed_values < lower) | (fixed_values > upper))).any()):
            raise ValueError("fixed values must lie inside their dimension bounds")

        fixed_integer = fixed_mask & integer_mask
        if bool(fixed_integer.any()):
            fixed_coordinates = (fixed_values[fixed_integer] - lower[fixed_integer]) / steps[
                fixed_integer
            ]
            if not bool(torch.isclose(fixed_coordinates, torch.round(fixed_coordinates)).all()):
                raise ValueError("fixed integer values must lie on their dimension step lattice")
        fixed_categorical = fixed_mask & categorical_mask
        if bool(fixed_categorical.any()) and not bool(
            torch.equal(
                fixed_values[fixed_categorical],
                torch.round(fixed_values[fixed_categorical]),
            )
        ):
            raise ValueError("fixed categorical values must be integer codes")

        categorical_bounds = torch.stack((lower[categorical_mask], upper[categorical_mask]))
        if categorical_bounds.numel() and not bool(
            torch.equal(categorical_bounds, torch.round(categorical_bounds))
        ):
            raise ValueError("categorical bounds must be integer-valued")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "integer_mask", integer_mask)
        object.__setattr__(self, "categorical_mask", categorical_mask)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "fixed_mask", fixed_mask)
        object.__setattr__(self, "fixed_values", fixed_values)
        mutable_mask = ~fixed_mask
        numeric_mask = ~categorical_mask
        mutable_numeric_mask = numeric_mask & mutable_mask
        mutable_integer_mask = integer_mask & mutable_mask
        mutable_categorical_mask = categorical_mask & mutable_mask
        object.__setattr__(self, "_numeric_mask", numeric_mask)
        object.__setattr__(self, "_mutable_numeric_mask", mutable_numeric_mask)
        object.__setattr__(self, "_mutable_integer_mask", mutable_integer_mask)
        object.__setattr__(self, "_mutable_categorical_mask", mutable_categorical_mask)
        object.__setattr__(self, "has_numeric", bool(mutable_numeric_mask.any()))
        object.__setattr__(self, "has_integer", bool(mutable_integer_mask.any()))
        object.__setattr__(self, "has_categorical", bool(mutable_categorical_mask.any()))
        object.__setattr__(self, "has_fixed", bool(fixed_mask.any()))
        object.__setattr__(self, "mutable_count", int(mutable_mask.sum().item()))

    @property
    def dimension(self) -> int:
        """Number of columns in a dense population."""

        return self.lower.numel()

    @property
    def numeric_mask(self) -> Tensor:
        """Mask of continuous and integer (but not categorical) columns."""

        return self._numeric_mask

    @property
    def mutable_numeric_mask(self) -> Tensor:
        """Mask of numeric columns that evolutionary operators may change."""

        return self._mutable_numeric_mask

    @property
    def mutable_integer_mask(self) -> Tensor:
        """Mask of integer columns that evolutionary operators may change."""

        return self._mutable_integer_mask

    @property
    def mutable_categorical_mask(self) -> Tensor:
        """Mask of categorical columns that evolutionary operators may change."""

        return self._mutable_categorical_mask


def repair_population(population: Tensor, spec: MixedVariableSpec) -> Tensor:
    """Clip and canonicalize a dense mixed-variable population.

    The returned tensor never aliases ``population``. Repair is intentionally idempotent: integer
    and stepped dimensions are rounded relative to their lower-bound anchor, categorical codes are
    rounded and bounded, and fixed values are written last so they remain exact.
    """

    if population.ndim != 2 or population.shape[1] != spec.dimension:
        raise ValueError(
            "population must have shape [n, d] with "
            f"d={spec.dimension}, got {tuple(population.shape)}"
        )
    if not population.is_floating_point():
        raise TypeError("population must use a floating-point dtype")
    if population.device != spec.lower.device or population.dtype != spec.lower.dtype:
        raise ValueError("population and search specification must share device and dtype")
    if not bool(torch.isfinite(population).all()):
        raise ValueError("population contains non-finite values")

    return _repair_population_unchecked(population, spec)


def _repair_population_unchecked(population: Tensor, spec: MixedVariableSpec) -> Tensor:
    """Repair an internally constructed population whose tensor contract is already known."""

    repaired = population.clamp(min=spec.lower, max=spec.upper)

    integer_mask = spec.mutable_integer_mask
    if spec.has_integer:
        lower = spec.lower[integer_mask]
        upper = spec.upper[integer_mask]
        steps = spec.steps[integer_mask]
        coordinates = torch.round((repaired[:, integer_mask] - lower) / steps)
        max_coordinates = torch.floor((upper - lower) / steps)
        coordinates = torch.maximum(coordinates, torch.zeros_like(coordinates))
        coordinates = torch.minimum(coordinates, max_coordinates)
        repaired[:, integer_mask] = coordinates * steps + lower

    categorical_mask = spec.mutable_categorical_mask
    if spec.has_categorical:
        repaired[:, categorical_mask] = torch.round(repaired[:, categorical_mask])

    repaired = repaired.clamp(min=spec.lower, max=spec.upper)
    if spec.has_fixed:
        repaired[:, spec.fixed_mask] = spec.fixed_values[spec.fixed_mask]
    return repaired
