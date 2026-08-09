# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.
"""Pure-Torch Box-Cox, Yeo-Johnson, and HEBO-style output transforms."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Protocol, TypeAlias, cast

import torch
from torch import Tensor, nn

PowerMethod: TypeAlias = Literal["box-cox", "yeo-johnson"]
OutputMethod: TypeAlias = Literal["auto", "none", "box-cox", "yeo-johnson"]

_METHOD_TO_CODE: dict[str, int] = {"none": 0, "box-cox": 1, "yeo-johnson": 2}
_STATE_SCHEMA_VERSION = 1


class _TypedStateModule(nn.Module):
    """Preserve checkpoint dtype/device when loading into an unfitted module."""

    _state_buffers: ClassVar[tuple[str, ...]] = ()
    _is_fitted: bool

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if not self._is_fitted:
            for name in self._state_buffers:
                incoming = state_dict.get(prefix + name)
                if isinstance(incoming, Tensor):
                    setattr(self, name, torch.empty_like(incoming))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class WarpConfigLike(Protocol):
    """Structural type accepted by :class:`OutputTransform`."""

    method: OutputMethod
    standardize_before_warp: bool
    refit_interval: int | None
    minimum_points: int
    minimum_transformed_std: float
    lambda_lower_bound: float
    lambda_upper_bound: float
    lambda_tolerance: float


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


def box_cox(x: Tensor, lmbda: float | Tensor) -> Tensor:
    """Apply the Box-Cox transform while preserving ``x``'s shape and device."""
    _require_float_tensor(x)
    invalid = torch.isfinite(x) & (x <= 0)
    if bool(invalid.any().item()):
        raise PowerTransformDomainError("Box-Cox requires every finite value to be positive")
    lam = _lambda_like(x, lmbda)
    log_x = torch.log(x)
    safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
    powered = torch.expm1(lam * log_x) / safe_lam
    return torch.where(lam == 0, log_x, powered)


def inverse_box_cox(y: Tensor, lmbda: float | Tensor) -> Tensor:
    """Invert :func:`box_cox`, raising on values outside the inverse domain."""
    _require_float_tensor(y)
    lam = _lambda_like(y, lmbda)
    base = 1 + lam * y
    invalid = torch.isfinite(y) & (lam != 0) & (base <= 0)
    if bool(invalid.any().item()):
        raise PowerTransformDomainError("inverse Box-Cox requires 1 + lambda * y > 0")
    safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
    powered = torch.exp(torch.log(base) / safe_lam)
    return torch.where(lam == 0, torch.exp(y), powered)


def yeo_johnson(x: Tensor, lmbda: float | Tensor) -> Tensor:
    """Apply the Yeo-Johnson transform to arbitrary real finite values."""
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


def inverse_yeo_johnson(y: Tensor, lmbda: float | Tensor) -> Tensor:
    """Invert :func:`yeo_johnson`, validating both branch domains."""
    _require_float_tensor(y)
    lam = _lambda_like(y, lmbda)
    nonnegative = y >= 0
    two_minus_lam = 2 - lam

    positive_y = torch.where(nonnegative, y, torch.zeros_like(y))
    negative_y = torch.where(nonnegative, torch.zeros_like(y), y)
    positive_base = 1 + lam * positive_y
    negative_base = 1 - two_minus_lam * negative_y
    invalid_positive = torch.isfinite(y) & nonnegative & (lam != 0) & (positive_base <= 0)
    invalid_negative = (
        torch.isfinite(y) & ~nonnegative & (two_minus_lam != 0) & (negative_base <= 0)
    )
    if bool((invalid_positive | invalid_negative).any().item()):
        raise PowerTransformDomainError("value lies outside the inverse Yeo-Johnson domain")

    safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
    positive_power = torch.expm1(torch.log(positive_base) / safe_lam)
    positive = torch.where(lam == 0, torch.expm1(y), positive_power)

    safe_two_minus_lam = torch.where(
        two_minus_lam == 0, torch.ones_like(two_minus_lam), two_minus_lam
    )
    negative_power = -torch.expm1(torch.log(negative_base) / safe_two_minus_lam)
    negative = torch.where(two_minus_lam == 0, -torch.expm1(-y), negative_power)
    return torch.where(nonnegative, positive, negative)


def power_transform(x: Tensor, lmbda: float | Tensor, method: PowerMethod) -> Tensor:
    """Apply one of the supported scalar-lambda power transforms."""
    normalized_method = _normalize_power_method(method)
    if normalized_method == "box-cox":
        return box_cox(x, lmbda)
    return yeo_johnson(x, lmbda)


def inverse_power_transform(y: Tensor, lmbda: float | Tensor, method: PowerMethod) -> Tensor:
    """Invert :func:`power_transform`."""
    normalized_method = _normalize_power_method(method)
    if normalized_method == "box-cox":
        return inverse_box_cox(y, lmbda)
    return inverse_yeo_johnson(y, lmbda)


def _normalize_power_method(method: str) -> PowerMethod:
    normalized = method.lower().replace("_", "-")
    if normalized not in {"box-cox", "yeo-johnson"}:
        raise ValueError(f"unsupported power-transform method: {method!r}")
    return cast(PowerMethod, normalized)


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
def fit_power_lambda(
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
    method = _normalize_power_method(method)
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


def fit_box_cox_lambda(
    x: Tensor,
    *,
    lower_bound: float = -5.0,
    upper_bound: float = 5.0,
    tolerance: float = 1e-5,
    grid_size: int = 33,
) -> Tensor:
    """Convenience wrapper around :func:`fit_power_lambda`."""
    return fit_power_lambda(
        x,
        "box-cox",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance=tolerance,
        grid_size=grid_size,
    )


def fit_yeo_johnson_lambda(
    x: Tensor,
    *,
    lower_bound: float = -5.0,
    upper_bound: float = 5.0,
    tolerance: float = 1e-5,
    grid_size: int = 33,
) -> Tensor:
    """Convenience wrapper around :func:`fit_power_lambda`."""
    return fit_power_lambda(
        x,
        "yeo-johnson",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance=tolerance,
        grid_size=grid_size,
    )


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


class PowerTransformer(_TypedStateModule):
    """A fitted one-dimensional Box-Cox or Yeo-Johnson transformer."""

    _lambda: Tensor
    _mean: Tensor
    _variance: Tensor
    _scale: Tensor
    _transform_version: Tensor
    _state_buffers = ("_lambda", "_mean", "_variance", "_scale", "_transform_version")

    def __init__(
        self,
        method: PowerMethod = "yeo-johnson",
        *,
        standardize: bool = True,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
        tolerance: float = 1e-5,
        grid_size: int = 33,
    ) -> None:
        super().__init__()
        self._method = _normalize_power_method(method)
        self.standardize = bool(standardize)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.tolerance = float(tolerance)
        self.grid_size = int(grid_size)
        self._is_fitted = False

        self.register_buffer("_lambda", torch.ones(()))
        self.register_buffer("_mean", torch.zeros(()))
        self.register_buffer("_variance", torch.ones(()))
        self.register_buffer("_scale", torch.ones(()))
        self.register_buffer("_transform_version", torch.zeros((), dtype=torch.int64))

    @property
    def method(self) -> PowerMethod:
        return self._method

    @property
    def fitted(self) -> bool:
        return self._is_fitted

    @property
    def lambda_(self) -> Tensor:
        return self._lambda

    @property
    def mean_(self) -> Tensor:
        return self._mean

    @property
    def var_(self) -> Tensor:
        return self._variance

    @property
    def scale_(self) -> Tensor:
        return self._scale

    @property
    def std(self) -> Tensor:
        return self._scale

    @property
    def version(self) -> int:
        return int(self._transform_version.item())

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("PowerTransformer must be fitted before use")

    @torch.no_grad()
    def fit(self, x: Tensor) -> PowerTransformer:
        _require_float_tensor(x)
        self.to(device=x.device, dtype=x.dtype)
        lmbda = fit_power_lambda(
            x,
            self.method,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            tolerance=self.tolerance,
            grid_size=self.grid_size,
        )
        transformed = power_transform(x.detach(), lmbda, self.method)
        mean, variance, scale, constant = _finite_summary(transformed.reshape(-1))
        if constant:
            raise PowerTransformFitError("transformed observations are numerically constant")

        self._lambda.copy_(lmbda)
        self._mean.copy_(mean if self.standardize else torch.zeros_like(mean))
        self._variance.copy_(variance)
        self._scale.copy_(scale if self.standardize else torch.ones_like(scale))
        self._transform_version.add_(1)
        self._is_fitted = True
        return self

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        self._require_fitted()
        transformed = power_transform(x, self._lambda, self.method)
        return (transformed - self._mean) / self._scale

    def inverse_transform(self, y: Tensor) -> Tensor:
        self._require_fitted()
        unstandardized = y * self._scale + self._mean
        return inverse_power_transform(unstandardized, self._lambda, self.method)

    def fit_transform(self, x: Tensor) -> Tensor:
        return self.fit(x).transform(x)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "method": self.method,
            "standardize": self.standardize,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "tolerance": self.tolerance,
            "grid_size": self.grid_size,
            "fitted": self.fitted,
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise RuntimeError("invalid PowerTransformer extra state")
        schema = int(state.get("schema_version", 0))
        if schema > _STATE_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported PowerTransformer state schema {schema}")
        self._method = _normalize_power_method(str(state.get("method", self.method)))
        self.standardize = bool(state.get("standardize", self.standardize))
        self.lower_bound = float(state.get("lower_bound", self.lower_bound))
        self.upper_bound = float(state.get("upper_bound", self.upper_bound))
        self.tolerance = float(state.get("tolerance", self.tolerance))
        self.grid_size = int(state.get("grid_size", self.grid_size))
        self._is_fitted = bool(state.get("fitted", True))


class _FromConfig:
    pass


_FROM_CONFIG = _FromConfig()


def _read_config(config: object | None, name: str, default: object) -> object:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


class OutputTransform(_TypedStateModule):
    """HEBO-compatible output standardization with an optional power warp.

    In ``auto`` mode, positive observations try Box-Cox first; observations that
    include zero or negative values use Yeo-Johnson. Numerically degenerate fits
    fall back to standardizing the unwarped target. Nonfinite entries are ignored
    during fitting and preserved by forward and inverse calls.
    """

    _lambda: Tensor
    _input_scale: Tensor
    _output_mean: Tensor
    _output_variance: Tensor
    _output_scale: Tensor
    _method_code: Tensor
    _transform_version: Tensor
    _raw_data_version: Tensor
    _standardizer_version: Tensor
    _warp_version: Tensor
    _last_refit_at: Tensor
    _state_buffers = (
        "_lambda",
        "_input_scale",
        "_output_mean",
        "_output_variance",
        "_output_scale",
        "_method_code",
        "_transform_version",
        "_raw_data_version",
        "_standardizer_version",
        "_warp_version",
        "_last_refit_at",
    )

    def __init__(
        self,
        config: WarpConfigLike | Mapping[str, object] | None = None,
        *,
        method: OutputMethod | None = None,
        standardize_before_warp: bool | None = None,
        refit_interval: int | _FromConfig | None = _FROM_CONFIG,
        minimum_points: int | None = None,
        minimum_transformed_std: float | None = None,
        lambda_lower_bound: float | None = None,
        lambda_upper_bound: float | None = None,
        lambda_tolerance: float | None = None,
        grid_size: int = 33,
    ) -> None:
        super().__init__()
        configured_method = str(_read_config(config, "method", "auto"))
        self._requested_method = _normalize_output_method(method or configured_method)
        configured_standardize = bool(_read_config(config, "standardize_before_warp", True))
        self.standardize_before_warp = (
            configured_standardize
            if standardize_before_warp is None
            else bool(standardize_before_warp)
        )
        configured_refit = _read_config(config, "refit_interval", 1)
        self.refit_interval = (
            cast(int | None, configured_refit)
            if isinstance(refit_interval, _FromConfig)
            else refit_interval
        )
        configured_minimum_points = _read_config(config, "minimum_points", 3)
        self.minimum_points = (
            int(minimum_points)
            if minimum_points is not None
            else int(cast(int, configured_minimum_points))
        )
        configured_minimum_std = _read_config(config, "minimum_transformed_std", 0.5)
        self.minimum_transformed_std = (
            float(minimum_transformed_std)
            if minimum_transformed_std is not None
            else float(cast(float, configured_minimum_std))
        )
        configured_lower_bound = _read_config(config, "lambda_lower_bound", -5.0)
        self.lambda_lower_bound = (
            float(lambda_lower_bound)
            if lambda_lower_bound is not None
            else float(cast(float, configured_lower_bound))
        )
        configured_upper_bound = _read_config(config, "lambda_upper_bound", 5.0)
        self.lambda_upper_bound = (
            float(lambda_upper_bound)
            if lambda_upper_bound is not None
            else float(cast(float, configured_upper_bound))
        )
        configured_tolerance = _read_config(config, "lambda_tolerance", 1e-5)
        self.lambda_tolerance = (
            float(lambda_tolerance)
            if lambda_tolerance is not None
            else float(cast(float, configured_tolerance))
        )
        self.grid_size = int(grid_size)
        self._validate_config()
        self._active_method: Literal["none", "box-cox", "yeo-johnson"] = "none"
        self._is_fitted = False

        self.register_buffer("_lambda", torch.ones(()))
        self.register_buffer("_input_scale", torch.ones(()))
        self.register_buffer("_output_mean", torch.zeros(()))
        self.register_buffer("_output_variance", torch.ones(()))
        self.register_buffer("_output_scale", torch.ones(()))
        self.register_buffer("_method_code", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_transform_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_raw_data_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_standardizer_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_warp_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_last_refit_at", torch.zeros((), dtype=torch.int64))

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
    def requested_method(self) -> OutputMethod:
        return self._requested_method

    @property
    def method(self) -> Literal["none", "box-cox", "yeo-johnson"]:
        """The method selected by the most recent successful fit."""
        return self._active_method

    @property
    def fitted(self) -> bool:
        return self._is_fitted

    @property
    def lambda_(self) -> Tensor:
        return self._lambda

    @property
    def input_scale(self) -> Tensor:
        return self._input_scale

    @property
    def mean_(self) -> Tensor:
        return self._output_mean

    @property
    def var_(self) -> Tensor:
        return self._output_variance

    @property
    def scale_(self) -> Tensor:
        return self._output_scale

    @property
    def std(self) -> Tensor:
        return self._output_scale

    @property
    def version(self) -> int:
        return int(self._transform_version.item())

    @property
    def raw_data_version(self) -> int:
        return int(self._raw_data_version.item())

    @property
    def output_standardizer_version(self) -> int:
        return int(self._standardizer_version.item())

    @property
    def output_warp_version(self) -> int:
        return int(self._warp_version.item())

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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
        if method == "none":
            lmbda = values.new_ones(())
            warped = values
        else:
            lmbda = fit_power_lambda(
                values,
                method,
                lower_bound=self.lambda_lower_bound,
                upper_bound=self.lambda_upper_bound,
                tolerance=self.lambda_tolerance,
                grid_size=self.grid_size,
            )
            warped = power_transform(values, lmbda, method)
        mean, variance, scale, constant = _finite_summary(warped)
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
        return lmbda, warped, mean, variance, scale, acceptable

    @torch.no_grad()
    def fit(self, y: Tensor, *, force: bool = False) -> OutputTransform:
        self._validate_objective(y)
        self.to(device=y.device, dtype=y.dtype)
        self._raw_data_version.add_(1)
        current_raw_version = int(self._raw_data_version.item())
        if self.fitted and not force:
            if self.refit_interval is None:
                return self
            last_refit = int(self._last_refit_at.item())
            if current_raw_version - last_refit < self.refit_interval:
                return self

        flat = y.detach().reshape(-1)
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

        requested = self.requested_method
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
        result: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, bool] | None = None
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

        lmbda, _, mean, variance, scale, _ = result
        self._lambda.copy_(lmbda)
        self._input_scale.copy_(input_scale)
        self._output_mean.copy_(mean)
        self._output_variance.copy_(variance)
        self._output_scale.copy_(scale)
        self._method_code.fill_(_METHOD_TO_CODE[selected])
        self._transform_version.add_(1)
        self._standardizer_version.add_(1)
        self._warp_version.add_(1)
        self._last_refit_at.fill_(current_raw_version)
        self._active_method = selected
        self._is_fitted = True
        return self

    def forward(self, y: Tensor) -> Tensor:
        return self.transform(y)

    def transform(self, y: Tensor) -> Tensor:
        self._require_fitted()
        self._validate_objective(y)
        finite = torch.isfinite(y)
        safe = torch.where(finite, y, torch.zeros_like(y)) / self._input_scale
        if self.method == "box-cox":
            # Invalid entries use a neutral positive placeholder and are restored below.
            safe = torch.where(finite, safe, torch.ones_like(safe))
            warped = box_cox(safe, self._lambda)
        elif self.method == "yeo-johnson":
            warped = yeo_johnson(safe, self._lambda)
        else:
            warped = safe
        transformed = (warped - self._output_mean) / self._output_scale
        return torch.where(finite, transformed, y)

    def inverse_transform(self, y: Tensor) -> Tensor:
        self._require_fitted()
        self._validate_objective(y)
        finite = torch.isfinite(y)
        unstandardized = y * self._output_scale + self._output_mean
        safe = torch.where(finite, unstandardized, torch.zeros_like(y))
        if self.method == "box-cox":
            unwarped = inverse_box_cox(safe, self._lambda)
        elif self.method == "yeo-johnson":
            unwarped = inverse_yeo_johnson(safe, self._lambda)
        else:
            unwarped = safe
        restored = unwarped * self._input_scale
        return torch.where(finite, restored, y)

    def fit_transform(self, y: Tensor, *, force: bool = False) -> Tensor:
        return self.fit(y, force=force).transform(y)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "requested_method": self.requested_method,
            "active_method": self.method,
            "standardize_before_warp": self.standardize_before_warp,
            "refit_interval": self.refit_interval,
            "minimum_points": self.minimum_points,
            "minimum_transformed_std": self.minimum_transformed_std,
            "lambda_lower_bound": self.lambda_lower_bound,
            "lambda_upper_bound": self.lambda_upper_bound,
            "lambda_tolerance": self.lambda_tolerance,
            "grid_size": self.grid_size,
            "fitted": self.fitted,
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise RuntimeError("invalid OutputTransform extra state")
        schema = int(state.get("schema_version", 0))
        if schema > _STATE_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported OutputTransform state schema {schema}")
        self._requested_method = _normalize_output_method(
            str(state.get("requested_method", self.requested_method))
        )
        active = _normalize_output_method(str(state.get("active_method", self.method)))
        if active == "auto":
            active = "none"
        self._active_method = active
        self.standardize_before_warp = bool(
            state.get("standardize_before_warp", self.standardize_before_warp)
        )
        refit = state.get("refit_interval", self.refit_interval)
        self.refit_interval = None if refit is None else int(refit)
        self.minimum_points = int(state.get("minimum_points", self.minimum_points))
        self.minimum_transformed_std = float(
            state.get("minimum_transformed_std", self.minimum_transformed_std)
        )
        self.lambda_lower_bound = float(state.get("lambda_lower_bound", self.lambda_lower_bound))
        self.lambda_upper_bound = float(state.get("lambda_upper_bound", self.lambda_upper_bound))
        self.lambda_tolerance = float(state.get("lambda_tolerance", self.lambda_tolerance))
        self.grid_size = int(state.get("grid_size", self.grid_size))
        self._is_fitted = bool(state.get("fitted", True))
        self._validate_config()


# Common spelling variants are useful at adapter boundaries.
boxcox = box_cox
inv_boxcox = inverse_box_cox
yeojohnson = yeo_johnson
inv_yeojohnson = inverse_yeo_johnson


__all__ = [
    "OutputMethod",
    "OutputTransform",
    "PowerMethod",
    "PowerTransformDomainError",
    "PowerTransformError",
    "PowerTransformFitError",
    "PowerTransformer",
    "WarpConfigLike",
    "box_cox",
    "boxcox",
    "fit_box_cox_lambda",
    "fit_power_lambda",
    "fit_yeo_johnson_lambda",
    "inv_boxcox",
    "inv_yeojohnson",
    "inverse_box_cox",
    "inverse_power_transform",
    "inverse_yeo_johnson",
    "power_transform",
    "yeo_johnson",
    "yeojohnson",
]
