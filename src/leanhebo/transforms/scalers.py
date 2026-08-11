# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.
"""Small, device-aware feature scalers implemented entirely in Torch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import Tensor, nn


def _require_float_batch(x: Tensor) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 2:
        raise ValueError(f"expected a two-dimensional tensor, got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"expected a floating-point tensor, got dtype {x.dtype}")
    if x.shape[1] == 0:
        raise ValueError("cannot fit a scaler with zero features")
    if x.shape[0] == 0:
        raise ValueError("cannot fit a scaler with zero observations")


class _ResizableBufferModule(nn.Module):
    """Allow feature-sized buffers to load into a newly constructed module."""

    _resizable_buffers: ClassVar[tuple[str, ...]] = ()

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
        unfitted = not bool(getattr(self, "fitted", False))
        for name in self._resizable_buffers:
            incoming = state_dict.get(prefix + name)
            if isinstance(incoming, Tensor):
                current = getattr(self, name)
                if not isinstance(current, Tensor) or current.shape != incoming.shape or unfitted:
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


class IdentityScaler(nn.Module):
    """No-op input scaling for spaces without continuous parameters."""

    @property
    def fitted(self) -> bool:
        """Identity scaling is always ready for use."""
        return True

    def fit(self, x: Tensor) -> IdentityScaler:
        if not isinstance(x, Tensor):
            raise TypeError(f"expected a torch.Tensor, got {type(x).__name__}")
        return self

    def transform(self, x: Tensor) -> Tensor:
        return x


class TorchMinMaxScaler(_ResizableBufferModule):
    """Map each finite feature to a configurable closed interval."""

    _resizable_buffers = (
        "data_min_",
        "data_max_",
        "scale_",
        "min_",
    )
    data_min_: Tensor
    data_max_: Tensor
    scale_: Tensor
    min_: Tensor

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        lower, upper = float(feature_range[0]), float(feature_range[1])
        if not lower < upper:
            raise ValueError("feature_range upper bound must be greater than its lower bound")

        self._feature_range = (lower, upper)
        self.register_buffer("data_min_", torch.empty(0))
        self.register_buffer("data_max_", torch.empty(0))
        self.register_buffer("scale_", torch.empty(0))
        self.register_buffer("min_", torch.empty(0))

    @property
    def fitted(self) -> bool:
        return self.data_min_.numel() > 0

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("TorchMinMaxScaler must be fitted before use")

    def _check_features(self, x: Tensor) -> None:
        self._require_fitted()
        features = self.data_min_.numel()
        if x.ndim == 0 or x.shape[-1] != features:
            raise ValueError(f"expected last dimension {features}, got shape {tuple(x.shape)}")

    @torch.no_grad()
    def fit(self, x: Tensor) -> TorchMinMaxScaler:
        _require_float_batch(x)
        self.to(device=x.device, dtype=x.dtype)
        values = x.detach()
        finite = torch.isfinite(values)
        count = finite.sum(dim=0)
        positive_inf = torch.full((), torch.inf, dtype=x.dtype, device=x.device)
        negative_inf = torch.full((), -torch.inf, dtype=x.dtype, device=x.device)
        data_min = torch.where(finite, values, positive_inf).amin(dim=0)
        data_max = torch.where(finite, values, negative_inf).amax(dim=0)
        valid = count > 0
        data_min = torch.where(valid, data_min, torch.zeros_like(data_min))
        data_max = torch.where(valid, data_max, torch.zeros_like(data_max))
        data_range = data_max - data_min

        lower, upper = self._feature_range
        near_constant = data_range.abs() <= (10 * torch.finfo(x.dtype).eps)
        denominator = torch.where(near_constant, torch.ones_like(data_range), data_range)
        scale = (upper - lower) / denominator
        offset = lower - data_min * scale

        # No-finite-data columns have no meaningful range; leave them neutral.
        scale = torch.where(valid, scale, torch.ones_like(scale))
        offset = torch.where(valid, offset, torch.zeros_like(offset))

        self.data_min_ = data_min
        self.data_max_ = data_max
        self.scale_ = scale
        self.min_ = offset
        return self

    def transform(self, x: Tensor) -> Tensor:
        self._check_features(x)
        return x * self.scale_ + self.min_


__all__ = [
    "IdentityScaler",
    "TorchMinMaxScaler",
]
