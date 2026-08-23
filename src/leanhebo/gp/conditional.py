# SPDX-License-Identifier: MIT

"""Persistent exact GP for compiled conditional parameter spaces."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any, cast

import gpytorch  # type: ignore[import-untyped]
import torch
from torch import nn

from leanhebo.config import GPConfig, RuntimeConfig
from leanhebo.data import EncodedBatch
from leanhebo.diagnostics import Diagnostics, FitReport
from leanhebo.errors import NumericalError
from leanhebo.gp.conditional_kernel import (
    ActivityFactorizedProductKernel,
    ActivityGroupSpec,
    ConditionalKernelLayout,
)
from leanhebo.gp.exact import ExactGPSurrogate
from leanhebo.space.compiled import CompiledSpace
from leanhebo.space.conditional import ConditionalSemantics


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


class _MaskedMinMaxScaler(nn.Module):
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
        for name in self._resizable_buffers:
            incoming = state_dict.get(prefix + name)
            if isinstance(incoming, torch.Tensor):
                current = getattr(self, name)
                if current.shape != incoming.shape:
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

    def would_change(self, x: torch.Tensor, activity: torch.Tensor) -> bool:
        """Return whether active appended values would change a fitted range."""

        if not self.fitted:
            return True
        mask = self._activity_mask(x, activity)
        known = self.active_count_ > 0
        outside = (~known)[None, :] | (x < self.data_min_) | (x > self.data_max_)
        return bool((mask & outside).any().item())


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
        self.num_continuous = space.n_continuous
        self.category_sizes = tuple(
            round(parameter.optimization_bounds[1] - parameter.optimization_bounds[0]) + 1
            for parameter in space.categorical_parameters
        )
        self.layout = _layout_from_semantics(semantics)
        self.layout.validate(
            num_continuous=self.num_continuous,
            num_categorical=len(self.category_sizes),
        )
        self.config = config
        self.runtime = runtime
        self.device = torch.device(runtime.device)
        self.dtype = getattr(torch, runtime.dtype)
        self.generator = generator
        self.diagnostics = diagnostics

        continuous_groups = torch.full((self.num_continuous,), -1, dtype=torch.int64)
        for group_index, group in enumerate(self.layout.groups):
            if group.continuous_indices:
                continuous_groups[list(group.continuous_indices)] = group_index
        continuous_lower = space.dense_lower_bounds[: self.num_continuous]
        continuous_upper = space.dense_upper_bounds[: self.num_continuous]
        self.input_scaler = cast(
            Any,
            _MaskedMinMaxScaler(continuous_groups, continuous_lower, continuous_upper),
        )
        self.input_scaler.to(device=self.device, dtype=self.dtype)
        self.model = None
        self.likelihood = None
        self.optimizer = None
        self.train_continuous = None
        self.train_categorical = None
        self.train_activity: torch.Tensor | None = None
        self.train_targets = None
        self.transform_version = -1
        self.update_count = 0
        self.full_refit_count = 0
        self.updates_since_full_refit = 0
        self.last_full_fit_observations = 0
        self.posterior_calls = 0

    def _settings(self) -> ExitStack:
        stack = super()._settings()
        stack.enter_context(gpytorch.settings.lazily_evaluate_kernels(False))
        return stack

    @property
    def masked_input_scaler(self) -> _MaskedMinMaxScaler:
        return cast(_MaskedMinMaxScaler, self.input_scaler)

    def _derive_activity(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        return self.semantics.activity(EncodedBatch(continuous, categorical)).group

    def _prepare_conditional_prediction_inputs(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous, categorical = self._coerce_inputs(continuous, categorical)
        activity = self._derive_activity(continuous, categorical)
        continuous = self.masked_input_scaler.transform(continuous, activity).contiguous()
        return continuous, categorical, activity

    def fit(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
        *,
        transform_version: int,
        force_full_refit: bool = False,
    ) -> FitReport:
        """Cold-fit, warm-update, or scheduled-refit the conditional exact GP."""

        continuous, categorical = self._coerce_inputs(continuous, categorical)
        activity = self._derive_activity(continuous, categorical)
        targets = torch.as_tensor(targets, device=self.device, dtype=self.dtype).reshape(-1)
        if targets.shape[0] != continuous.shape[0]:
            raise ValueError("target and input batch lengths differ")
        if not torch.isfinite(targets).all():
            raise ValueError("GP targets must all be finite")
        if targets.numel() < 2:
            raise ValueError("at least two observations are required to fit an exact GP")

        first_fit = self.model is None
        full_refit = first_fit or force_full_refit or self._full_refit_due(targets.numel())
        kind = "initial" if first_fit else ("full_refit" if full_refit else "update")
        steps = self.config.initial_steps if full_refit else self.config.update_steps
        previous_model_state = self.model.state_dict() if self.model is not None else None
        previous_likelihood_state = (
            self.likelihood.state_dict() if self.likelihood is not None else None
        )
        previous_optimizer_state = (
            self.optimizer.state_dict() if self.optimizer is not None else None
        )
        transform_changed = self.transform_version != transform_version

        if self.config.use_fantasy_updates and not full_refit:
            fantasy_start = time.perf_counter()
            skip_reason = self._try_conditional_fantasy_update(
                continuous,
                categorical,
                activity,
                targets,
                transform_version=transform_version,
                optimizer_state=previous_optimizer_state,
            )
            if skip_reason is None:
                self.transform_version = transform_version
                report = FitReport(
                    kind="fantasy_update",
                    observations=targets.numel(),
                    requested_steps=0,
                    completed_steps=0,
                    final_loss=None,
                    wall_time=time.perf_counter() - fantasy_start,
                )
                self.update_count += 1
                self.updates_since_full_refit += 1
                if self.diagnostics is not None:
                    self.diagnostics.add_fit_report(report)
                    self.diagnostics.increment("gp.fantasy_update")
                return report
            if self.diagnostics is not None:
                self.diagnostics.increment("gp.fantasy_skipped")
                self.diagnostics.increment(f"gp.fantasy_skipped.{skip_reason}")

        self.masked_input_scaler.fit(continuous, activity)
        continuous = self.masked_input_scaler.transform(continuous, activity).contiguous()
        self.train_continuous = continuous
        self.train_categorical = categorical
        self.train_activity = activity
        self.train_targets = targets
        self.transform_version = transform_version

        if full_refit or not self.config.use_set_train_data or not self.config.reuse_parameters:
            self._construct_conditional_model(
                retain_model_state=(
                    previous_model_state if self.config.reuse_parameters and not first_fit else None
                ),
                retain_likelihood_state=(
                    previous_likelihood_state
                    if self.config.reuse_parameters and not first_fit
                    else None
                ),
                retain_optimizer_state=(
                    previous_optimizer_state
                    if (
                        self.config.reuse_optimizer_state
                        and not first_fit
                        and (not full_refit or not self.config.reset_optimizer_on_full_refit)
                    )
                    else None
                ),
                initialize_kernel=first_fit or not self.config.reuse_parameters,
            )
        else:
            model = cast(_ConditionalExactGP, self.model)
            model.set_train_data(
                inputs=(continuous, categorical, activity),
                targets=targets,
                strict=False,
            )
            if not self.config.reuse_optimizer_state:
                self.optimizer = self._create_optimizer()

        if transform_changed and self.diagnostics is not None:
            self.diagnostics.increment("gp.transform_invalidations")
        report = self._optimize(kind, steps)
        if first_fit or full_refit:
            self.full_refit_count += 1
            self.updates_since_full_refit = 0
            self.last_full_fit_observations = targets.numel()
        else:
            self.update_count += 1
            self.updates_since_full_refit += 1
        if self.diagnostics is not None:
            self.diagnostics.add_fit_report(report)
            self.diagnostics.increment(f"gp.{kind}")
        return report

    @torch.no_grad()
    def _try_conditional_fantasy_update(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        activity: torch.Tensor,
        targets: torch.Tensor,
        *,
        transform_version: int,
        optimizer_state: Mapping[str, Any] | None,
    ) -> str | None:
        if (
            self.model is None
            or self.likelihood is None
            or self.optimizer is None
            or self.train_continuous is None
            or self.train_categorical is None
            or self.train_activity is None
            or self.train_targets is None
        ):
            return "missing_state"
        model = cast(_ConditionalExactGP, self.model)
        if model.prediction_strategy is None:
            return "missing_prediction_cache"
        if transform_version != self.transform_version:
            return "output_transform_changed"

        old_count = self.train_targets.numel()
        if targets.numel() <= old_count:
            return "not_append_only"
        if (
            continuous.shape[0] != targets.numel()
            or categorical.shape[0] != targets.numel()
            or activity.shape[0] != targets.numel()
            or self.train_continuous.shape[0] != old_count
            or self.train_categorical.shape[0] != old_count
            or self.train_activity.shape[0] != old_count
        ):
            return "not_append_only"

        new_continuous = continuous[old_count:]
        new_activity = activity[old_count:]
        if self.masked_input_scaler.would_change(new_continuous, new_activity):
            return "input_transform_changed"
        scaled_continuous = self.masked_input_scaler.transform(continuous, activity).contiguous()
        if not torch.equal(scaled_continuous[:old_count], self.train_continuous):
            return "input_transform_changed"
        if not torch.equal(categorical[:old_count], self.train_categorical):
            return "not_append_only"
        if not torch.equal(activity[:old_count], self.train_activity):
            return "not_append_only"
        if not torch.equal(targets[:old_count], self.train_targets):
            return "output_transform_changed"

        old_prediction_strategy = model.prediction_strategy
        old_train_inputs = model.train_inputs
        old_train_targets = model.train_targets
        old_model_likelihood = model.likelihood
        try:
            with (
                self._settings(),
                gpytorch.settings.fast_pred_var(self.config.fast_pred_var),
            ):
                fantasy_model = model.get_fantasy_model(
                    [
                        scaled_continuous[old_count:],
                        categorical[old_count:],
                        activity[old_count:],
                    ],
                    targets[old_count:],
                )
        finally:
            model.prediction_strategy = old_prediction_strategy
            model.train_inputs = old_train_inputs
            model.train_targets = old_train_targets
            model.likelihood = old_model_likelihood

        self.model = cast(Any, fantasy_model)
        self.likelihood = fantasy_model.likelihood
        self.train_continuous = scaled_continuous
        self.train_categorical = categorical
        self.train_activity = activity
        self.train_targets = targets
        self.optimizer = self._create_optimizer()
        if self.config.reuse_optimizer_state and optimizer_state is not None:
            self.optimizer.load_state_dict(dict(optimizer_state))
        return None

    def _construct_conditional_model(
        self,
        *,
        retain_model_state: Mapping[str, Any] | None,
        retain_likelihood_state: Mapping[str, Any] | None,
        retain_optimizer_state: Mapping[str, Any] | None,
        initialize_kernel: bool,
    ) -> None:
        assert self.train_continuous is not None
        assert self.train_categorical is not None
        assert self.train_activity is not None
        assert self.train_targets is not None
        module_seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (),
                device=self.generator.device,
                generator=self.generator,
            ).item()
        )
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(module_seed)
            constraint = gpytorch.constraints.GreaterThan(self.config.noise_lower_bound)
            prior = gpytorch.priors.LogNormalPrior(math.log(self.config.noise_initial), 0.5)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=constraint,
                noise_prior=prior,
            ).to(device=self.device, dtype=self.dtype)
            model = _ConditionalExactGP(
                self.train_continuous,
                self.train_categorical,
                self.train_activity,
                self.train_targets,
                likelihood,
                category_sizes=self.category_sizes,
                layout=self.layout,
                ard=self.config.ard,
            ).to(device=self.device, dtype=self.dtype)
        likelihood.noise = max(self.config.noise_initial, self.config.noise_lower_bound)
        if initialize_kernel:
            model.activity_kernel.initialize_numeric_lengthscales(
                self.train_continuous,
                self.train_activity,
                sample_limit=self.config.kernel_initialization_samples,
                lower_bound=self.config.lengthscale_lower_bound,
                generator=self.generator,
            )
            variance = self.train_targets.var(unbiased=False).clamp_min(torch.finfo(self.dtype).eps)
            model.covar_module.outputscale = variance
        if retain_model_state is not None:
            model.load_state_dict(retain_model_state)
        if retain_likelihood_state is not None:
            likelihood.load_state_dict(retain_likelihood_state)
        self.model = cast(Any, model)
        self.likelihood = likelihood
        self.optimizer = self._create_optimizer()
        if retain_optimizer_state is not None:
            self.optimizer.load_state_dict(dict(retain_optimizer_state))

    def _training_loss(
        self,
        mll: gpytorch.mlls.ExactMarginalLogLikelihood,
    ) -> torch.Tensor:
        assert self.model is not None
        assert self.train_continuous is not None
        assert self.train_categorical is not None
        assert self.train_activity is not None
        assert self.train_targets is not None
        model = cast(_ConditionalExactGP, self.model)
        with self._settings():
            distribution = model(
                self.train_continuous,
                self.train_categorical,
                self.train_activity,
            )
            loss: torch.Tensor = -mll(distribution, self.train_targets)
        return loss

    @torch.no_grad()
    def predict(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mean, variance, and scalar observation-noise variance."""

        if self.model is None or self.likelihood is None:
            raise RuntimeError("the GP has not been fitted")
        continuous, categorical, activity = self._prepare_conditional_prediction_inputs(
            continuous, categorical
        )
        model = cast(_ConditionalExactGP, self.model)
        if model.training:
            model.eval()
        if self.likelihood.training:
            self.likelihood.eval()
        self.posterior_calls += 1
        if self.diagnostics is not None:
            self.diagnostics.increment("posterior.calls")
            self.diagnostics.increment("posterior.candidates", continuous.shape[0])
        with (
            self._settings(),
            gpytorch.settings.fast_pred_var(self.config.fast_pred_var),
        ):
            if self.config.eval_cg_tolerance is None:
                distribution = model(continuous, categorical, activity)
            else:
                with gpytorch.settings.eval_cg_tolerance(self.config.eval_cg_tolerance):
                    distribution = model(continuous, categorical, activity)
            if self.config.predict_observation_noise:
                distribution = self.likelihood(distribution)
        mean = distribution.mean.reshape(-1)
        variance = distribution.variance.reshape(-1).clamp_min(torch.finfo(self.dtype).eps)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise NumericalError("conditional exact-GP posterior contains non-finite values")
        return mean, variance, self.noise_variance

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_scaler": self.input_scaler.state_dict(),
            "model": None if self.model is None else self.model.state_dict(),
            "likelihood": None if self.likelihood is None else self.likelihood.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "train_continuous": self.train_continuous,
            "train_categorical": self.train_categorical,
            "train_activity": self.train_activity,
            "train_targets": self.train_targets,
            "transform_version": self.transform_version,
            "updates_since_full_refit": self.updates_since_full_refit,
            "last_full_fit_observations": self.last_full_fit_observations,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        scaler_state = state["input_scaler"]
        if not isinstance(scaler_state, Mapping):
            raise ValueError("conditional exact-GP input scaler state is missing or malformed")
        self.input_scaler.load_state_dict(dict(scaler_state))
        self.input_scaler.to(device=self.device, dtype=self.dtype)
        train_continuous = state["train_continuous"]
        train_categorical = state["train_categorical"]
        train_activity = state["train_activity"]
        train_targets = state["train_targets"]
        if train_continuous is not None:
            if train_categorical is None or train_activity is None or train_targets is None:
                raise ValueError("conditional exact-GP training state is incomplete")
            self.train_continuous = torch.as_tensor(
                train_continuous,
                device=self.device,
                dtype=self.dtype,
            )
            self.train_categorical = torch.as_tensor(
                train_categorical,
                device=self.device,
                dtype=torch.int64,
            )
            self.train_activity = torch.as_tensor(
                train_activity,
                device=self.device,
                dtype=torch.bool,
            )
            self.train_targets = torch.as_tensor(
                train_targets,
                device=self.device,
                dtype=self.dtype,
            )
            self._construct_conditional_model(
                retain_model_state=state["model"],
                retain_likelihood_state=state["likelihood"],
                retain_optimizer_state=state["optimizer"],
                initialize_kernel=False,
            )
            assert self.model is not None and self.likelihood is not None
            self.model.eval()
            self.likelihood.eval()
        self.transform_version = int(state["transform_version"])
        self.updates_since_full_refit = int(state["updates_since_full_refit"])
        self.last_full_fit_observations = int(state["last_full_fit_observations"])


__all__ = ["ConditionalExactGPSurrogate"]
