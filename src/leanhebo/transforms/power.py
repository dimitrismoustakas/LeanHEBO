# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.
"""Pure-Torch Box-Cox, Yeo-Johnson, and HEBO-style output transforms."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor, nn

from leanhebo.config import WarpConfig

PowerMethod: TypeAlias = Literal["box-cox", "yeo-johnson"]
OutputMethod: TypeAlias = Literal["auto", "none", "box-cox", "yeo-johnson"]


class PowerTransformError(ValueError):
    """Base class for invalid or numerically degenerate power transforms."""


class PowerTransformDomainError(PowerTransformError):
    """Raised when values lie outside a transform's mathematical domain."""


class PowerTransformFitError(PowerTransformError):
    """Raised when a lambda cannot be estimated from the supplied observations."""


def _require_float_tensor(x: Tensor) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(x).__name__}")
    if not x.is_floating_point():
        raise TypeError(f"expected a floating-point tensor, got dtype {x.dtype}")


def _lambda_like(x: Tensor, lmbda: float | Tensor) -> Tensor:
    if isinstance(lmbda, Tensor):
        if lmbda.numel() != 1:
            raise ValueError("lambda must be scalar")
        return lmbda.to(dtype=x.dtype, device=x.device).reshape(())
    return x.new_tensor(float(lmbda))


def _box_cox(x: Tensor, lmbda: float | Tensor) -> Tensor:
    _require_float_tensor(x)
    invalid = torch.isfinite(x) & (x <= 0)
    if bool(invalid.any().item()):
        raise PowerTransformDomainError("Box-Cox requires every finite value to be positive")
    lam = _lambda_like(x, lmbda)
    log_x = torch.log(x)
    safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
    powered = torch.expm1(lam * log_x) / safe_lam
    return torch.where(lam == 0, log_x, powered)


def _yeo_johnson(x: Tensor, lmbda: float | Tensor) -> Tensor:
    _require_float_tensor(x)
    lam = _lambda_like(x, lmbda)
    nonnegative = x >= 0

    positive_x = torch.where(nonnegative, x, torch.zeros_like(x))
    positive_log = torch.log1p(positive_x)
    safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
    positive_power = torch.expm1(lam * positive_log) / safe_lam
    positive = torch.where(lam == 0, positive_log, positive_power)

    negative_magnitude = torch.where(nonnegative, torch.zeros_like(x), -x)
    negative_log = torch.log1p(negative_magnitude)
    two_minus_lam = 2 - lam
    safe_two_minus_lam = torch.where(
        two_minus_lam == 0, torch.ones_like(two_minus_lam), two_minus_lam
    )
    negative_power = -torch.expm1(two_minus_lam * negative_log) / safe_two_minus_lam
    negative = torch.where(two_minus_lam == 0, -negative_log, negative_power)
    return torch.where(nonnegative, positive, negative)


def _normalize_output_method(method: str) -> OutputMethod:
    normalized = method.lower().replace("_", "-")
    if normalized not in {"auto", "none", "box-cox", "yeo-johnson"}:
        raise ValueError(f"unsupported output-transform method: {method!r}")
    return cast(OutputMethod, normalized)


def _power_matrix(x: Tensor, lambdas: Tensor, method: PowerMethod) -> Tensor:
    """Vectorized transform for a ``[lambda, observation]`` likelihood grid."""
    lam = lambdas.reshape(-1, 1)
    values = x.reshape(1, -1)
    if method == "box-cox":
        log_values = torch.log(values)
        safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
        transformed = torch.expm1(lam * log_values) / safe_lam
        return torch.where(lam == 0, log_values.expand_as(transformed), transformed)

    nonnegative = values >= 0
    positive_values = torch.where(nonnegative, values, torch.zeros_like(values))
    positive_log = torch.log1p(positive_values)
    safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
    positive_power = torch.expm1(lam * positive_log) / safe_lam
    positive = torch.where(lam == 0, positive_log.expand_as(positive_power), positive_power)

    negative_values = torch.where(nonnegative, torch.zeros_like(values), -values)
    negative_log = torch.log1p(negative_values)
    two_minus_lam = 2 - lam
    safe_two_minus_lam = torch.where(
        two_minus_lam == 0, torch.ones_like(two_minus_lam), two_minus_lam
    )
    negative_power = -torch.expm1(two_minus_lam * negative_log) / safe_two_minus_lam
    negative = torch.where(
        two_minus_lam == 0,
        -negative_log.expand_as(negative_power),
        negative_power,
    )
    return torch.where(nonnegative, positive, negative)


def _profile_likelihood(x: Tensor, lambdas: Tensor, method: PowerMethod) -> Tensor:
    transformed = _power_matrix(x, lambdas, method)
    variance = transformed.var(dim=1, correction=0)
    n = x.numel()
    if method == "box-cox":
        jacobian_sum = torch.log(x).sum()
    else:
        jacobian_sum = (torch.sign(x) * torch.log1p(torch.abs(x))).sum()
    likelihood = -0.5 * n * torch.log(variance)
    likelihood = likelihood + (lambdas - 1) * jacobian_sum
    valid = torch.isfinite(likelihood) & torch.isfinite(variance) & (variance > 0)
    return torch.where(valid, likelihood, torch.full_like(likelihood, -torch.inf))


@torch.no_grad()
def _fit_power_lambda(
    x: Tensor,
    method: PowerMethod,
    *,
    lower_bound: float = -5.0,
    upper_bound: float = 5.0,
    tolerance: float = 1e-5,
    grid_size: int = 33,
) -> Tensor:
    """Maximum-likelihood lambda fit using a grid and bounded golden search.

    The returned scalar tensor stays on the input device and uses the input dtype.
    Nonfinite observations are excluded from fitting.
    """
    _require_float_tensor(x)
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
        raise ValueError("lambda bounds must be finite")
    if not lower_bound < upper_bound:
        raise ValueError("lambda upper bound must be greater than its lower bound")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("lambda tolerance must be a positive finite number")
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")

    values = x.detach().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() < 2:
        raise PowerTransformFitError("at least two finite observations are required")
    if method == "box-cox" and bool((values <= 0).any().item()):
        raise PowerTransformDomainError("Box-Cox fitting requires positive observations")
    if bool((values == values[0]).all().item()):
        raise PowerTransformFitError("lambda is undefined for constant observations")

    grid = torch.linspace(
        lower_bound,
        upper_bound,
        grid_size,
        dtype=values.dtype,
        device=values.device,
    )
    scores = _profile_likelihood(values, grid, method)
    if not bool(torch.isfinite(scores).any().item()):
        raise PowerTransformFitError("profile likelihood is nonfinite for every lambda")
    best_index = int(torch.argmax(scores).item())
    left_index = max(0, best_index - 1)
    right_index = min(grid_size - 1, best_index + 1)
    left = float(grid[left_index].item())
    right = float(grid[right_index].item())

    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def score(value: float) -> float:
        candidate = values.new_tensor([value])
        return float(_profile_likelihood(values, candidate, method)[0].item())

    c = right - golden_ratio * (right - left)
    d = left + golden_ratio * (right - left)
    score_c = score(c)
    score_d = score(d)
    while right - left > tolerance:
        if score_c > score_d:
            right = d
            d = c
            score_d = score_c
            c = right - golden_ratio * (right - left)
            score_c = score(c)
        else:
            left = c
            c = d
            score_c = score_d
            d = left + golden_ratio * (right - left)
            score_d = score(d)

    candidates = [lower_bound, upper_bound, left, right, c, d, (left + right) / 2]
    candidate_tensor = values.new_tensor(candidates)
    candidate_scores = _profile_likelihood(values, candidate_tensor, method)
    result = candidate_tensor[torch.argmax(candidate_scores)]
    if not bool(torch.isfinite(result).item()):
        raise PowerTransformFitError("lambda optimization produced a nonfinite result")
    return result.detach()


def _finite_summary(values: Tensor) -> tuple[Tensor, Tensor, Tensor, bool]:
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        zero = values.new_zeros(())
        one = values.new_ones(())
        return zero, zero, one, True
    mean = finite_values.mean()
    variance = finite_values.var(correction=0)
    count = values.new_tensor(float(finite_values.numel()))
    eps = torch.finfo(values.dtype).eps
    upper_bound = count * eps * variance + (count * mean * eps).square()
    constant = bool((variance <= upper_bound).item())
    scale = torch.where(
        variance <= upper_bound,
        torch.ones_like(variance),
        torch.sqrt(variance),
    )
    return mean, variance, scale, constant


class OutputTransform(nn.Module):
    """HEBO-compatible output standardization with an optional power warp.

    In ``auto`` mode, positive observations try Box-Cox first; observations that
    include zero or negative values use Yeo-Johnson. Numerically degenerate fits
    fall back to standardizing the unwarped target. Nonfinite entries are ignored
    during fitting and preserved by transformation.
    """

    _lambda: Tensor
    _input_scale: Tensor
    _output_mean: Tensor
    _output_scale: Tensor
    _transform_version: Tensor
    _fit_calls: Tensor
    _last_refit_call: Tensor

    def __init__(self, config: WarpConfig | None = None) -> None:
        super().__init__()
        config = config or WarpConfig()
        self._requested_method = _normalize_output_method(config.method)
        self.standardize_before_warp = config.standardize_before_warp
        self.refit_interval = config.refit_interval
        self.minimum_points = config.minimum_points
        self.minimum_transformed_std = config.minimum_transformed_std
        self.lambda_lower_bound = config.lambda_lower_bound
        self.lambda_upper_bound = config.lambda_upper_bound
        self.lambda_tolerance = config.lambda_tolerance
        self.grid_size = 33
        self._validate_config()
        self._active_method: Literal["none", "box-cox", "yeo-johnson"] = "none"
        self._is_fitted = False

        self.register_buffer("_lambda", torch.ones(()))
        self.register_buffer("_input_scale", torch.ones(()))
        self.register_buffer("_output_mean", torch.zeros(()))
        self.register_buffer("_output_scale", torch.ones(()))
        self.register_buffer("_transform_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_fit_calls", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_last_refit_call", torch.zeros((), dtype=torch.int64))

    def _validate_config(self) -> None:
        if self.refit_interval is not None and self.refit_interval < 1:
            raise ValueError("refit_interval must be positive or None")
        if self.minimum_points < 1:
            raise ValueError("minimum_points must be positive")
        if not math.isfinite(self.minimum_transformed_std) or self.minimum_transformed_std < 0:
            raise ValueError("minimum_transformed_std must be finite and nonnegative")
        if not self.lambda_lower_bound < self.lambda_upper_bound:
            raise ValueError("lambda upper bound must be greater than its lower bound")
        if not math.isfinite(self.lambda_tolerance) or self.lambda_tolerance <= 0:
            raise ValueError("lambda_tolerance must be positive and finite")
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least three")

    @property
    def method(self) -> Literal["none", "box-cox", "yeo-johnson"]:
        """The method selected by the most recent successful fit."""
        return self._active_method

    @property
    def fitted(self) -> bool:
        return self._is_fitted

    @property
    def version(self) -> int:
        return int(self._transform_version.item())

    @staticmethod
    def _validate_objective(y: Tensor) -> None:
        _require_float_tensor(y)
        if y.ndim not in {1, 2} or (y.ndim == 2 and y.shape[1] != 1):
            raise ValueError(f"expected objective shape [n] or [n, 1], got {tuple(y.shape)}")
        if y.numel() == 0:
            raise ValueError("cannot fit an empty objective tensor")

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("OutputTransform must be fitted before use")

    def _candidate(
        self,
        values: Tensor,
        method: Literal["none", "box-cox", "yeo-johnson"],
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        if method == "none":
            lmbda = values.new_ones(())
            warped = values
        else:
            lmbda = _fit_power_lambda(
                values,
                method,
                lower_bound=self.lambda_lower_bound,
                upper_bound=self.lambda_upper_bound,
                tolerance=self.lambda_tolerance,
                grid_size=self.grid_size,
            )
            warped = _box_cox(values, lmbda) if method == "box-cox" else _yeo_johnson(values, lmbda)
        mean, _, scale, constant = _finite_summary(warped)
        standardized = (warped - mean) / scale
        finite_standardized = standardized[torch.isfinite(standardized)]
        transformed_std = (
            finite_standardized.var(correction=0).sqrt()
            if finite_standardized.numel() > 0
            else values.new_zeros(())
        )
        acceptable = (
            not constant
            and bool(torch.isfinite(transformed_std).item())
            and float(transformed_std.item()) >= self.minimum_transformed_std
        )
        return lmbda, mean, scale, acceptable

    @torch.no_grad()
    def fit(self, y: Tensor, *, force: bool = False) -> OutputTransform:
        self._validate_objective(y)
        self.to(device=y.device, dtype=y.dtype)
        self._fit_calls.add_(1)
        fit_call = int(self._fit_calls.item())
        if self.fitted and not force:
            if self.refit_interval is None:
                return self
            last_refit = int(self._last_refit_call.item())
            if fit_call - last_refit < self.refit_interval:
                return self

        # This is a tiny scalar fit; CPU execution avoids a synchronization per golden step.
        flat = y.detach().reshape(-1).cpu()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            input_scale = flat.new_ones(())
            normalized = finite
            raw_constant = True
        else:
            _, _, finite_scale, raw_constant = _finite_summary(finite)
            input_scale = (
                finite_scale
                if self.standardize_before_warp and not raw_constant
                else flat.new_ones(())
            )
            normalized = finite / input_scale

        requested = self._requested_method
        if requested == "auto":
            if finite.numel() < self.minimum_points or raw_constant:
                candidates: tuple[Literal["none", "box-cox", "yeo-johnson"], ...] = ("none",)
            elif bool((normalized > 0).all().item()):
                candidates = ("box-cox", "yeo-johnson", "none")
            else:
                candidates = ("yeo-johnson", "none")
        else:
            candidates = (requested,)

        selected: Literal["none", "box-cox", "yeo-johnson"] | None = None
        result: tuple[Tensor, Tensor, Tensor, bool] | None = None
        last_error: PowerTransformError | None = None
        for candidate in candidates:
            try:
                candidate_result = self._candidate(normalized, candidate)
            except PowerTransformError as error:
                last_error = error
                if requested != "auto":
                    raise
                continue
            if candidate == "none" or candidate_result[-1]:
                selected = candidate
                result = candidate_result
                break
            if requested != "auto":
                raise PowerTransformFitError(
                    f"{candidate} produced transformed std below minimum_transformed_std"
                )

        if selected is None or result is None:
            if last_error is not None:
                raise PowerTransformFitError("unable to fit output transform") from last_error
            raise PowerTransformFitError("unable to fit output transform")

        lmbda, mean, scale, _ = result
        self._lambda.copy_(lmbda)
        self._input_scale.copy_(input_scale)
        self._output_mean.copy_(mean)
        self._output_scale.copy_(scale)
        self._transform_version.add_(1)
        self._last_refit_call.fill_(fit_call)
        self._active_method = selected
        self._is_fitted = True
        return self

    def transform(self, y: Tensor) -> Tensor:
        self._require_fitted()
        self._validate_objective(y)
        finite = torch.isfinite(y)
        safe = torch.where(finite, y, torch.zeros_like(y)) / self._input_scale
        if self.method == "box-cox":
            # Invalid entries use a neutral positive placeholder and are restored below.
            safe = torch.where(finite, safe, torch.ones_like(safe))
            warped = _box_cox(safe, self._lambda)
        elif self.method == "yeo-johnson":
            warped = _yeo_johnson(safe, self._lambda)
        else:
            warped = safe
        transformed = (warped - self._output_mean) / self._output_scale
        return torch.where(finite, transformed, y)

    def fit_transform(self, y: Tensor, *, force: bool = False) -> Tensor:
        return self.fit(y, force=force).transform(y)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "active_method": self.method,
            "fitted": self.fitted,
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise RuntimeError("invalid OutputTransform extra state")
        active = _normalize_output_method(str(state["active_method"]))
        if active == "auto":
            raise RuntimeError("invalid active OutputTransform method")
        self._active_method = active
        self._is_fitted = bool(state["fitted"])


__all__ = [
    "OutputTransform",
    "PowerTransformDomainError",
    "PowerTransformError",
    "PowerTransformFitError",
]
