# SPDX-License-Identifier: MIT

"""Persistent exact GP for compiled conditional parameter spaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any, cast

import gpytorch  # type: ignore[import-untyped]
import torch

from leanhebo.config import GPConfig, RuntimeConfig
from leanhebo.data import EncodedBatch
from leanhebo.diagnostics import Diagnostics
from leanhebo.gp.conditional_kernel import (
    ActivityFactorizedProductKernel,
    ActivityGroupSpec,
    ConditionalKernelLayout,
)
from leanhebo.gp.exact import ExactGPSurrogate
from leanhebo.space.compiled import CompiledSpace
from leanhebo.space.conditional import ConditionalSemantics
from leanhebo.transforms.scalers import _ResizableBufferModule


def _layout_from_semantics(semantics: ConditionalSemantics) -> ConditionalKernelLayout:
    continuous_columns = {
        parameter_index: column
        for column, parameter_index in enumerate(semantics.continuous_parameter_indices)
    }
    categorical_columns = {
        parameter_index: column
        for column, parameter_index in enumerate(semantics.categorical_parameter_indices)
    }
    root_continuous = tuple(
        column
        for parameter_index, column in continuous_columns.items()
        if semantics.parameter_to_group[parameter_index] < 0
    )
    root_categorical = tuple(
        column
        for parameter_index, column in categorical_columns.items()
        if semantics.parameter_to_group[parameter_index] < 0
    )
    groups = tuple(
        ActivityGroupSpec(
            tuple(
                continuous_columns[index]
                for index in parameter_indices
                if index in continuous_columns
            ),
            tuple(
                categorical_columns[index]
                for index in parameter_indices
                if index in categorical_columns
            ),
        )
        for parameter_indices in semantics.group_parameter_indices
    )
    return ConditionalKernelLayout(root_continuous, root_categorical, groups)


class _MaskedMinMaxScaler(_ResizableBufferModule):
    """Scale continuous columns from active observations and zero inactive values."""

    _resizable_buffers = ("data_min_", "data_max_", "scale_", "min_", "active_count_")
    continuous_groups: torch.Tensor
    data_min_: torch.Tensor
    data_max_: torch.Tensor
    scale_: torch.Tensor
    min_: torch.Tensor
    active_count_: torch.Tensor
    _fitted: torch.Tensor
    _is_fitted: bool
    conditional_indices: torch.Tensor
    conditional_groups: torch.Tensor
    domain_lower: torch.Tensor
    domain_upper: torch.Tensor

    def __init__(
        self,
        continuous_groups: torch.Tensor,
        domain_lower: torch.Tensor,
        domain_upper: torch.Tensor,
    ) -> None:
        super().__init__()
        if continuous_groups.ndim != 1 or continuous_groups.dtype != torch.int64:
            raise TypeError("continuous activity groups must be a one-dimensional int64 tensor")
        if (
            domain_lower.ndim != 1
            or domain_upper.ndim != 1
            or domain_lower.shape != continuous_groups.shape
            or domain_upper.shape != continuous_groups.shape
        ):
            raise ValueError("continuous domain bounds have an incompatible shape")
        if not torch.isfinite(domain_lower).all() or not torch.isfinite(domain_upper).all():
            raise ValueError("continuous domain bounds must be finite")
        if bool((domain_lower >= domain_upper).any()):
            raise ValueError("continuous domain lower bounds must be strictly below upper bounds")
        self.register_buffer("continuous_groups", continuous_groups.clone(), persistent=False)
        conditional = continuous_groups >= 0
        self.register_buffer(
            "conditional_indices",
            torch.nonzero(conditional, as_tuple=False).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "conditional_groups",
            continuous_groups[conditional],
            persistent=False,
        )
        self.register_buffer("domain_lower", domain_lower.clone(), persistent=False)
        self.register_buffer("domain_upper", domain_upper.clone(), persistent=False)
        self.register_buffer("data_min_", torch.empty(0))
        self.register_buffer("data_max_", torch.empty(0))
        self.register_buffer("scale_", torch.empty(0))
        self.register_buffer("min_", torch.empty(0))
        self.register_buffer("active_count_", torch.empty(0, dtype=torch.int64))
        self.register_buffer("_fitted", torch.tensor(False))
        self._is_fitted = False

    @property
    def fitted(self) -> bool:
        return self._is_fitted

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._is_fitted = bool(self._fitted.item())

    def _activity_mask(self, x: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, self.conditional_indices] = activity.index_select(1, self.conditional_groups)
        return mask

    @torch.no_grad()
    def fit(self, x: torch.Tensor, activity: torch.Tensor) -> _MaskedMinMaxScaler:
        mask = self._activity_mask(x, activity)
        self.to(device=x.device, dtype=x.dtype)
        count = mask.sum(dim=0)
        positive_inf = torch.full((), torch.inf, dtype=x.dtype, device=x.device)
        negative_inf = torch.full((), -torch.inf, dtype=x.dtype, device=x.device)
        data_min = torch.where(mask, x, positive_inf).amin(dim=0)
        data_max = torch.where(mask, x, negative_inf).amax(dim=0)
        valid = count > 0
        data_min = torch.where(valid, data_min, self.domain_lower)
        data_max = torch.where(valid, data_max, self.domain_upper)
        data_range = data_max - data_min
        near_constant = data_range.abs() <= (10 * torch.finfo(x.dtype).eps)
        denominator = torch.where(near_constant, torch.ones_like(data_range), data_range)
        scale = 2.0 / denominator
        offset = -1.0 - data_min * scale
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.scale_ = scale
        self.min_ = offset
        self.active_count_ = count
        self._fitted.fill_(True)
        self._is_fitted = True
        return self

    def transform(self, x: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("masked GP input scaler has not been fitted")
        mask = self._activity_mask(x, activity)
        return torch.where(mask, x * self.scale_ + self.min_, torch.zeros_like(x))


class _ConditionalExactGP(gpytorch.models.ExactGP):  # type: ignore[misc]
    def __init__(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        activity: torch.Tensor,
        targets: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        category_sizes: Sequence[int],
        layout: ConditionalKernelLayout,
        ard: bool,
    ) -> None:
        super().__init__((continuous, categorical, activity), targets, likelihood)
        base = ActivityFactorizedProductKernel(
            category_sizes=category_sizes,
            layout=layout,
            ard=ard,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base,
            outputscale_prior=gpytorch.priors.GammaPrior(0.5, 0.5),
        )

    @property
    def activity_kernel(self) -> ActivityFactorizedProductKernel:
        return cast(ActivityFactorizedProductKernel, self.covar_module.base_kernel)

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        activity: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        packed = self.activity_kernel.pack_inputs(continuous, categorical, activity)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(packed), self.covar_module(packed)
        )


class ConditionalExactGPSurrogate(ExactGPSurrogate):
    """Exact GP whose covariance factors through compiled conditional semantics."""

    def __init__(
        self,
        *,
        space: CompiledSpace,
        config: GPConfig,
        runtime: RuntimeConfig,
        generator: torch.Generator,
        diagnostics: Diagnostics | None = None,
    ) -> None:
        semantics = space.conditional_semantics
        if semantics is None:
            raise ValueError("ConditionalExactGPSurrogate requires a conditional compiled space")
        self.space = space
        self.semantics = semantics
        self.layout = _layout_from_semantics(semantics)
        category_sizes = tuple(
            round(parameter.optimization_bounds[1] - parameter.optimization_bounds[0]) + 1
            for parameter in space.categorical_parameters
        )
        self.layout.validate(
            num_continuous=space.n_continuous,
            num_categorical=len(category_sizes),
        )
        super().__init__(
            num_continuous=space.n_continuous,
            category_sizes=category_sizes,
            config=config,
            runtime=runtime,
            generator=generator,
            diagnostics=diagnostics,
        )
        continuous_groups = torch.full((self.num_continuous,), -1, dtype=torch.int64)
        for group_index, group in enumerate(self.layout.groups):
            if group.continuous_indices:
                continuous_groups[list(group.continuous_indices)] = group_index
        self.input_scaler = cast(
            Any,
            _MaskedMinMaxScaler(
                continuous_groups,
                space.dense_lower_bounds[: self.num_continuous],
                space.dense_upper_bounds[: self.num_continuous],
            ),
        )
        self.input_scaler.to(device=self.device, dtype=self.dtype)
        self.train_activity: torch.Tensor | None = None

    @property
    def masked_input_scaler(self) -> _MaskedMinMaxScaler:
        return cast(_MaskedMinMaxScaler, self.input_scaler)

    def _settings(self) -> ExitStack:
        stack = super()._settings()
        stack.enter_context(gpytorch.settings.lazily_evaluate_kernels(False))
        return stack

    def _derive_activity(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        return self.semantics.activity(EncodedBatch(continuous, categorical)).group

    def _store_training_data(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        activity = self._derive_activity(continuous, categorical)
        self.masked_input_scaler.fit(continuous, activity)
        self.train_continuous = self.masked_input_scaler.transform(
            continuous, activity
        ).contiguous()
        self.train_categorical = categorical
        self.train_activity = activity
        self.train_targets = targets

    def _train_inputs(self) -> tuple[torch.Tensor, ...]:
        assert self.train_activity is not None
        return (*super()._train_inputs(), self.train_activity)

    def _prepare_prediction_inputs(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        *,
        validate: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        continuous, categorical = self._coerce_inputs(
            continuous,
            categorical,
            validate=validate,
        )
        activity = self._derive_activity(continuous, categorical)
        continuous = self.masked_input_scaler.transform(continuous, activity).contiguous()
        return continuous, categorical, activity

    def _build_model(
        self, likelihood: gpytorch.likelihoods.GaussianLikelihood
    ) -> gpytorch.models.ExactGP:
        assert self.train_continuous is not None
        assert self.train_categorical is not None
        assert self.train_activity is not None
        assert self.train_targets is not None
        return _ConditionalExactGP(
            self.train_continuous,
            self.train_categorical,
            self.train_activity,
            self.train_targets,
            likelihood,
            category_sizes=self.category_sizes,
            layout=self.layout,
            ard=self.config.ard,
        )

    def _initialize_kernel(self, model: gpytorch.models.ExactGP) -> None:
        assert self.train_continuous is not None
        assert self.train_activity is not None
        model.activity_kernel.initialize_numeric_lengthscales(
            self.train_continuous,
            self.train_activity,
            sample_limit=self.config.kernel_initialization_samples,
            lower_bound=self.config.lengthscale_lower_bound,
            generator=self.generator,
        )

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "train_activity": self.train_activity}

    def _restore_extra_training_state(self, state: Mapping[str, Any]) -> None:
        train_activity = state["train_activity"]
        if train_activity is None:
            raise ValueError("conditional exact-GP training state is incomplete")
        self.train_activity = torch.as_tensor(
            train_activity,
            device=self.device,
            dtype=torch.bool,
        )


__all__ = ["ConditionalExactGPSurrogate"]
