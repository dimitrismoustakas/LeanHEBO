# SPDX-License-Identifier: MIT

"""Activity-factorized covariance for compiled conditional parameter spaces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import gpytorch  # type: ignore[import-untyped]
import torch
import torch.nn.functional as functional
from torch import nn

from leanhebo.gp.kernel import (
    MixedFeatureExtractor,
    build_base_kernel,
    initialize_base_numeric_lengthscales,
)

_INITIAL_ALPHA = 0.2
_ACTIVITY_LOGIT_PRIOR_SCALE = 1.0
_INITIAL_ACTIVITY_LOGIT = math.log(_INITIAL_ALPHA / (1.0 - _INITIAL_ALPHA))


@dataclass(frozen=True, slots=True)
class ActivityGroupSpec:
    """Global GP-input columns sharing one compiled activity expression."""

    continuous_indices: tuple[int, ...] = ()
    categorical_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "continuous_indices", tuple(int(index) for index in self.continuous_indices)
        )
        object.__setattr__(
            self, "categorical_indices", tuple(int(index) for index in self.categorical_indices)
        )


@dataclass(frozen=True, slots=True)
class ConditionalKernelLayout:
    """Partition GP columns into an always-active root and conditional groups."""

    root_continuous_indices: tuple[int, ...]
    root_categorical_indices: tuple[int, ...]
    groups: tuple[ActivityGroupSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_continuous_indices",
            tuple(int(index) for index in self.root_continuous_indices),
        )
        object.__setattr__(
            self,
            "root_categorical_indices",
            tuple(int(index) for index in self.root_categorical_indices),
        )
        object.__setattr__(self, "groups", tuple(self.groups))
        if not self.groups:
            raise ValueError("a conditional kernel layout requires at least one activity group")

    def validate(self, *, num_continuous: int, num_categorical: int) -> None:
        """Require each GP input column to belong to exactly one feature block."""

        continuous = [*self.root_continuous_indices]
        categorical = [*self.root_categorical_indices]
        for group in self.groups:
            continuous.extend(group.continuous_indices)
            categorical.extend(group.categorical_indices)
        if sorted(continuous) != list(range(num_continuous)):
            raise ValueError("conditional continuous columns must form an exact partition")
        if sorted(categorical) != list(range(num_categorical)):
            raise ValueError("conditional categorical columns must form an exact partition")


class _UnitKernel(gpytorch.kernels.Kernel):  # type: ignore[misc]
    """Parameter-free multiplicative identity for an empty feature block."""

    is_stationary = True

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        diag: bool = False,
        **params: object,
    ) -> torch.Tensor:
        del params
        if diag:
            return x1.new_ones(x1.shape[-2])
        return x1.new_ones((x1.shape[-2], x2.shape[-2]))


class _FeatureBlock(nn.Module):
    """Select and embed the numeric/categorical inputs owned by one activity block."""

    continuous_indices: torch.Tensor
    categorical_indices: torch.Tensor
    kernel: gpytorch.kernels.Kernel

    def __init__(
        self,
        *,
        continuous_indices: Sequence[int],
        categorical_indices: Sequence[int],
        category_sizes: Sequence[int],
        ard: bool,
    ) -> None:
        super().__init__()
        continuous_indices = tuple(int(index) for index in continuous_indices)
        categorical_indices = tuple(int(index) for index in categorical_indices)
        self.register_buffer(
            "continuous_indices",
            torch.tensor(continuous_indices, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "categorical_indices",
            torch.tensor(categorical_indices, dtype=torch.int64),
            persistent=False,
        )
        selected_sizes = tuple(int(category_sizes[index]) for index in categorical_indices)
        self.feature_extractor = MixedFeatureExtractor(len(continuous_indices), selected_sizes)
        if self.feature_extractor.output_dimensions:
            self.kernel: gpytorch.kernels.Kernel = build_base_kernel(
                num_continuous=len(continuous_indices),
                feature_extractor=self.feature_extractor,
                ard=ard,
            )
        else:
            self.kernel = _UnitKernel()

    @property
    def output_dimensions(self) -> int:
        return self.feature_extractor.output_dimensions

    def features(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        continuous_indices = self.continuous_indices.to(device=continuous.device)
        categorical_indices = self.categorical_indices.to(device=categorical.device)
        selected_continuous = continuous.index_select(1, continuous_indices)
        selected_categorical = categorical.index_select(1, categorical_indices)
        result: torch.Tensor = self.feature_extractor(selected_continuous, selected_categorical)
        return result


def _to_dense(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    to_dense = getattr(value, "to_dense", None)
    if to_dense is None:
        raise TypeError("kernel evaluation did not return a tensor or linear operator")
    dense = to_dense()
    if not isinstance(dense, torch.Tensor):
        raise TypeError("kernel linear operator did not materialize to a tensor")
    return dense


class ActivityFactorizedProductKernel(gpytorch.kernels.Kernel):  # type: ignore[misc]
    """Unit-diagonal product of one normalized factor per activity group."""

    is_stationary = False
    activity_logit: torch.Tensor

    def __init__(
        self,
        *,
        category_sizes: Sequence[int],
        layout: ConditionalKernelLayout,
        ard: bool,
    ) -> None:
        super().__init__()
        self.root_block = _FeatureBlock(
            continuous_indices=layout.root_continuous_indices,
            categorical_indices=layout.root_categorical_indices,
            category_sizes=category_sizes,
            ard=ard,
        )
        self.group_blocks = nn.ModuleList(
            _FeatureBlock(
                continuous_indices=group.continuous_indices,
                categorical_indices=group.categorical_indices,
                category_sizes=category_sizes,
                ard=ard,
            )
            for group in layout.groups
        )
        self.activity_logit = nn.Parameter(
            torch.full((len(layout.groups),), _INITIAL_ACTIVITY_LOGIT)
        )
        self.register_prior(
            "activity_logit_prior",
            gpytorch.priors.NormalPrior(
                _INITIAL_ACTIVITY_LOGIT,
                _ACTIVITY_LOGIT_PRIOR_SCALE,
            ),
            "activity_logit",
        )

        offset = 0
        self._root_slice = slice(offset, offset + self.root_block.output_dimensions)
        offset = self._root_slice.stop
        group_slices: list[slice] = []
        for module in self.group_blocks:
            block = cast(_FeatureBlock, module)
            group_slice = slice(offset, offset + block.output_dimensions)
            group_slices.append(group_slice)
            offset += block.output_dimensions
        self._group_slices = tuple(group_slices)
        self._activity_slice = slice(offset, offset + len(layout.groups))
        self.packed_dimensions = self._activity_slice.stop

    @property
    def batch_shape(self) -> torch.Size:
        """Return the kernel's fixed unbatched shape without traversing its feature blocks."""

        return torch.Size()

    @property
    def alpha(self) -> torch.Tensor:
        """Local active-function variance fractions."""

        log_beta = functional.logsigmoid(-self.activity_logit)
        return -torch.expm1(log_beta)

    @property
    def beta(self) -> torch.Tensor:
        """Shared structural variance fractions, computed without cancellation."""

        return functional.logsigmoid(-self.activity_logit).exp()

    @property
    def rho(self) -> torch.Tensor:
        """Active/inactive correlations."""

        return (0.5 * functional.logsigmoid(-self.activity_logit)).exp()

    def pack_inputs(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        """Embed mixed blocks and append Boolean activity bits for covariance evaluation."""

        features = [self.root_block.features(continuous, categorical)]
        features.extend(
            cast(_FeatureBlock, block).features(continuous, categorical)
            for block in self.group_blocks
        )
        features.append(activity.to(device=continuous.device, dtype=continuous.dtype))
        return torch.cat(features, dim=-1)

    def initialize_numeric_lengthscales(
        self,
        continuous: torch.Tensor,
        activity: torch.Tensor,
        *,
        sample_limit: int,
        lower_bound: float,
        generator: torch.Generator,
    ) -> None:
        """Initialize each numeric block from rows where that complete block is active."""

        root_values = continuous.index_select(
            1, self.root_block.continuous_indices.to(device=continuous.device)
        )
        initialize_base_numeric_lengthscales(
            self.root_block.kernel,
            root_values,
            sample_limit=sample_limit,
            lower_bound=lower_bound,
            generator=generator,
        )
        for group_index, module in enumerate(self.group_blocks):
            block = cast(_FeatureBlock, module)
            active_values = continuous[activity[:, group_index]].index_select(
                1, block.continuous_indices.to(device=continuous.device)
            )
            initialize_base_numeric_lengthscales(
                block.kernel,
                active_values,
                sample_limit=sample_limit,
                lower_bound=lower_bound,
                generator=generator,
            )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        diag: bool = False,
        **params: object,
    ) -> torch.Tensor:
        del params
        root = self.root_block.kernel(x1[:, self._root_slice], x2[:, self._root_slice], diag=diag)
        covariance = _to_dense(root)
        activity1 = x1[:, self._activity_slice] > 0.5
        activity2 = x2[:, self._activity_slice] > 0.5
        alpha = self.alpha
        beta = self.beta
        rho = self.rho

        for group_index, (module, feature_slice) in enumerate(
            zip(self.group_blocks, self._group_slices, strict=True)
        ):
            block = cast(_FeatureBlock, module)
            active1 = activity1[:, group_index]
            active2 = activity2[:, group_index]
            if x1.device.type == "cuda":
                local = _to_dense(
                    block.kernel(
                        x1[:, feature_slice],
                        x2[:, feature_slice],
                        diag=diag,
                    )
                )
                if diag:
                    both = active1 & active2
                    mismatch = torch.logical_xor(active1, active2)
                else:
                    both = active1[:, None] & active2[None, :]
                    mismatch = torch.logical_xor(active1[:, None], active2[None, :])
                active_factor = beta[group_index] + alpha[group_index] * local
                factor = torch.where(
                    both,
                    active_factor,
                    torch.where(mismatch, rho[group_index], torch.ones_like(covariance)),
                )
                covariance = covariance * factor
                continue
            if diag:
                mismatch = torch.logical_xor(active1, active2)
                factor = torch.where(mismatch, rho[group_index], torch.ones_like(covariance))
                both = active1 & active2
                indices = both.nonzero(as_tuple=False).reshape(-1)
                if indices.numel():
                    local = block.kernel(
                        x1.index_select(0, indices)[:, feature_slice],
                        x2.index_select(0, indices)[:, feature_slice],
                        diag=True,
                    )
                    active_factor = beta[group_index] + alpha[group_index] * _to_dense(local)
                    factor = factor.index_put((indices,), active_factor)
            else:
                mismatch = torch.logical_xor(active1[:, None], active2[None, :])
                factor = torch.where(mismatch, rho[group_index], torch.ones_like(covariance))
                indices1 = active1.nonzero(as_tuple=False).reshape(-1)
                indices2 = active2.nonzero(as_tuple=False).reshape(-1)
                if indices1.numel() and indices2.numel():
                    local = block.kernel(
                        x1.index_select(0, indices1)[:, feature_slice],
                        x2.index_select(0, indices2)[:, feature_slice],
                    )
                    active_factor = beta[group_index] + alpha[group_index] * _to_dense(local)
                    factor = factor.index_put(
                        (indices1[:, None], indices2[None, :]),
                        active_factor,
                    )
            covariance = covariance * factor
        return covariance


__all__ = [
    "ActivityFactorizedProductKernel",
    "ActivityGroupSpec",
    "ConditionalKernelLayout",
]
