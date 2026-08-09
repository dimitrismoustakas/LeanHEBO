# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Persistent HEBO-compatible exact Gaussian-process surrogate."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any, cast

import gpytorch  # type: ignore[import-untyped]
import torch

from leanhebo.config import GPConfig, RuntimeConfig
from leanhebo.diagnostics import Diagnostics, FitReport
from leanhebo.errors import NumericalError
from leanhebo.gp.kernel import (
    MixedFeatureExtractor,
    build_kernel,
    initialize_numeric_lengthscales,
)
from leanhebo.gp.optimizer import PreconditionedSGLD, create_optimizer
from leanhebo.transforms import IdentityScaler, TorchMinMaxScaler

_STATE_SCHEMA_VERSION = 2


class _MixedExactGP(gpytorch.models.ExactGP):  # type: ignore[misc]
    def __init__(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        category_sizes: Sequence[int],
        ard: bool,
    ) -> None:
        super().__init__((continuous, categorical), targets, likelihood)
        self.feature_extractor = MixedFeatureExtractor(continuous.shape[1], category_sizes)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = build_kernel(
            num_continuous=continuous.shape[1],
            feature_extractor=self.feature_extractor,
            ard=ard,
        )

    def forward(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        features = self.feature_extractor(continuous, categorical)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(features), self.covar_module(features)
        )


class ExactGPSurrogate:
    """Own an exact GP, likelihood, optimizer, training state, and prediction lifecycle."""

    def __init__(
        self,
        *,
        num_continuous: int,
        category_sizes: Sequence[int],
        config: GPConfig,
        runtime: RuntimeConfig,
        generator: torch.Generator,
        diagnostics: Diagnostics | None = None,
    ) -> None:
        if num_continuous < 0:
            raise ValueError("num_continuous cannot be negative")
        if num_continuous + len(category_sizes) == 0:
            raise ValueError("the surrogate requires at least one non-fixed dimension")
        self.num_continuous = num_continuous
        self.category_sizes = tuple(int(size) for size in category_sizes)
        self.config = config
        self.runtime = runtime
        self.device = torch.device(runtime.device)
        self.dtype = getattr(torch, runtime.dtype)
        self.generator = generator
        self.diagnostics = diagnostics
        self.input_scaler: IdentityScaler | TorchMinMaxScaler
        if self.num_continuous:
            self.input_scaler = TorchMinMaxScaler(feature_range=(-1.0, 1.0))
        else:
            self.input_scaler = IdentityScaler()
        self.input_scaler.to(device=self.device, dtype=self.dtype)
        self.model: _MixedExactGP | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.train_continuous: torch.Tensor | None = None
        self.train_categorical: torch.Tensor | None = None
        self.train_targets: torch.Tensor | None = None
        self.transform_version = -1
        self.fit_count = 0
        self.update_count = 0
        self.full_refit_count = 0
        self.updates_since_full_refit = 0
        self.last_full_fit_observations = 0
        self.posterior_calls = 0
        self.parameter_version = 0
        self.posterior_cache_version = 0

    @property
    def fitted(self) -> bool:
        return self.model is not None

    @property
    def noise_variance(self) -> torch.Tensor:
        if self.likelihood is None:
            raise RuntimeError("the GP has not been fitted")
        noise: torch.Tensor = self.likelihood.noise.detach().reshape(())
        return noise

    @property
    def input_scaler_version(self) -> int:
        """Version of the observed-data input scaling used by the active GP."""

        return self.input_scaler.version

    def _coerce_inputs(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        continuous = torch.as_tensor(continuous, device=self.device, dtype=self.dtype)
        categorical = torch.as_tensor(categorical, device=self.device, dtype=torch.int64)
        if continuous.ndim != 2 or continuous.shape[1] != self.num_continuous:
            raise ValueError("continuous input shape is incompatible with the surrogate")
        if categorical.ndim != 2 or categorical.shape[1] != len(self.category_sizes):
            raise ValueError("categorical input shape is incompatible with the surrogate")
        if continuous.shape[0] != categorical.shape[0]:
            raise ValueError("continuous and categorical batch lengths differ")
        if not torch.isfinite(continuous).all():
            raise ValueError("continuous GP inputs must all be finite")
        for index, size in enumerate(self.category_sizes):
            if torch.any((categorical[:, index] < 0) | (categorical[:, index] >= size)):
                raise ValueError(f"categorical codes in dimension {index} are out of bounds")
        return continuous.contiguous(), categorical.contiguous()

    def _prepare_prediction_inputs(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        continuous, categorical = self._coerce_inputs(continuous, categorical)
        if self.num_continuous and not self.input_scaler.fitted:
            raise RuntimeError("the GP input scaler has not been fitted")
        return self.input_scaler.transform(continuous).contiguous(), categorical

    def fit(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
        *,
        transform_version: int,
        force_full_refit: bool = False,
        reset_parameters: bool = False,
    ) -> FitReport:
        """Cold-fit, warm-update, or scheduled-refit the exact GP."""

        continuous, categorical = self._coerce_inputs(continuous, categorical)
        targets = torch.as_tensor(targets, device=self.device, dtype=self.dtype).reshape(-1)
        if targets.shape[0] != continuous.shape[0]:
            raise ValueError("target and input batch lengths differ")
        if not torch.isfinite(targets).all():
            raise ValueError("GP targets must all be finite")
        if targets.numel() < 2:
            raise ValueError("at least two observations are required to fit an exact GP")

        first_fit = self.model is None
        full_refit = (
            first_fit
            or force_full_refit
            or reset_parameters
            or self._full_refit_due(targets.numel())
        )
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
            fallback_reason = self._try_fantasy_update(
                continuous,
                categorical,
                targets,
                transform_version=transform_version,
                optimizer_state=previous_optimizer_state,
            )
            if fallback_reason is None:
                self.transform_version = transform_version
                self.posterior_cache_version += 1
                report = FitReport(
                    kind="fantasy_update",
                    observations=targets.numel(),
                    requested_steps=0,
                    completed_steps=0,
                    final_loss=None,
                    wall_time=time.perf_counter() - fantasy_start,
                    maximum_jitter=self.config.jitter_initial,
                    jitter_retries=0,
                )
                self.fit_count += 1
                self.update_count += 1
                self.updates_since_full_refit += 1
                if self.diagnostics is not None:
                    self.diagnostics.add_fit_report(report)
                    self.diagnostics.increment("gp.fantasy_update")
                return report
            if self.diagnostics is not None:
                self.diagnostics.increment("gp.fantasy_fallback")
                self.diagnostics.increment(f"gp.fantasy_fallback.{fallback_reason}")

        previous_input_scaler_version = self.input_scaler_version
        self.input_scaler.fit(continuous)
        continuous = self.input_scaler.transform(continuous).contiguous()

        self.train_continuous = continuous
        self.train_categorical = categorical
        self.train_targets = targets
        self.transform_version = transform_version

        if full_refit or not self.config.use_set_train_data or not self.config.reuse_parameters:
            self._construct_model(
                retain_model_state=(
                    previous_model_state
                    if self.config.reuse_parameters and not first_fit and not reset_parameters
                    else None
                ),
                retain_likelihood_state=(
                    previous_likelihood_state
                    if self.config.reuse_parameters and not first_fit and not reset_parameters
                    else None
                ),
                retain_optimizer_state=(
                    previous_optimizer_state
                    if self.config.reuse_optimizer_state and not first_fit and not reset_parameters
                    else None
                ),
                initialize_kernel=first_fit or reset_parameters or not self.config.reuse_parameters,
            )
        else:
            assert self.model is not None
            self.model.set_train_data(
                inputs=(continuous, categorical), targets=targets, strict=False
            )
            if not self.config.reuse_optimizer_state:
                self.optimizer = self._create_optimizer()
            elif isinstance(self.optimizer, PreconditionedSGLD):
                self.optimizer.set_observation_count(targets.numel())

        self.posterior_cache_version += 1
        if previous_input_scaler_version and self.diagnostics is not None:
            self.diagnostics.increment("gp.input_transform_invalidations")
        if transform_changed and self.diagnostics is not None:
            self.diagnostics.increment("gp.transform_invalidations")
        report = self._optimize(kind, steps)
        self.fit_count += 1
        if first_fit or full_refit:
            self.full_refit_count += 1
            self.updates_since_full_refit = 0
            self.last_full_fit_observations = targets.numel()
        else:
            self.update_count += 1
            self.updates_since_full_refit += 1
        self.parameter_version += int(report.completed_steps > 0)
        if self.diagnostics is not None:
            self.diagnostics.add_fit_report(report)
            self.diagnostics.increment(f"gp.{kind}")
        return report

    def _full_refit_due(self, observations: int) -> bool:
        if self.model is None:
            return True
        interval = self.config.full_refit_interval
        if interval is not None and self.updates_since_full_refit + 1 >= interval:
            return True
        growth = self.config.full_refit_growth_factor
        return bool(
            growth is not None
            and self.last_full_fit_observations > 0
            and observations >= math.ceil(self.last_full_fit_observations * growth)
        )

    @torch.no_grad()
    def _try_fantasy_update(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
        *,
        transform_version: int,
        optimizer_state: Mapping[str, Any] | None,
    ) -> str | None:
        """Append data through GPyTorch's exact cache update, or return a fallback reason."""

        if (
            self.model is None
            or self.likelihood is None
            or self.optimizer is None
            or self.train_continuous is None
            or self.train_categorical is None
            or self.train_targets is None
        ):
            return "missing_state"
        if self.model.prediction_strategy is None:
            return "missing_prediction_cache"
        if transform_version != self.transform_version:
            return "output_transform_changed"

        old_count = self.train_targets.numel()
        if targets.numel() <= old_count:
            return "not_append_only"
        appended_count = targets.numel() - old_count
        if appended_count >= old_count:
            # GPyTorch's fantasy update is intended for a small append relative to the cached
            # history. Rebuilding is the safer cost profile once the append is as large as it.
            return "batch_not_smaller_than_history"
        if (
            continuous.shape[0] != targets.numel()
            or categorical.shape[0] != targets.numel()
            or self.train_continuous.shape[0] != old_count
            or self.train_categorical.shape[0] != old_count
        ):
            return "not_append_only"

        new_continuous = continuous[old_count:]
        if self.num_continuous:
            scaler = cast(TorchMinMaxScaler, self.input_scaler)
            if bool(
                ((new_continuous < scaler.data_min_) | (new_continuous > scaler.data_max_))
                .any()
                .item()
            ):
                return "input_transform_changed"
        scaled_continuous = self.input_scaler.transform(continuous).contiguous()
        if not torch.equal(scaled_continuous[:old_count], self.train_continuous):
            return "input_transform_changed"
        if not torch.equal(categorical[:old_count], self.train_categorical):
            return "not_append_only"
        if not torch.equal(targets[:old_count], self.train_targets):
            return "output_transform_changed"

        old_prediction_strategy = self.model.prediction_strategy
        old_train_inputs = self.model.train_inputs
        old_train_targets = self.model.train_targets
        old_model_likelihood = self.model.likelihood
        try:
            with (
                self._settings(self.config.jitter_initial),
                gpytorch.settings.fast_pred_var(self.config.fast_pred_var),
            ):
                fantasy_model = self.model.get_fantasy_model(
                    [scaled_continuous[old_count:], categorical[old_count:]],
                    targets[old_count:],
                )
        except RuntimeError as error:
            if not _looks_numerical(error):
                raise
            return "numerical_error"
        finally:
            # GPyTorch temporarily removes these fields while deep-copying an ExactGP but does
            # not protect that mutation with its own finally block.
            self.model.prediction_strategy = old_prediction_strategy
            self.model.train_inputs = old_train_inputs
            self.model.train_targets = old_train_targets
            self.model.likelihood = old_model_likelihood

        self.model = cast(_MixedExactGP, fantasy_model)
        self.likelihood = cast(gpytorch.likelihoods.GaussianLikelihood, fantasy_model.likelihood)
        self.train_continuous = scaled_continuous
        self.train_categorical = categorical
        self.train_targets = targets
        optimizer = self._create_optimizer()
        if self.config.reuse_optimizer_state and optimizer_state is not None:
            try:
                optimizer.load_state_dict(dict(optimizer_state))
            except (KeyError, TypeError, ValueError, RuntimeError):
                if self.diagnostics is not None:
                    self.diagnostics.increment("gp.optimizer_state_incompatible")
        if isinstance(optimizer, PreconditionedSGLD):
            optimizer.set_observation_count(targets.numel())
        self.optimizer = optimizer
        return None

    def _construct_model(
        self,
        *,
        retain_model_state: Mapping[str, Any] | None,
        retain_likelihood_state: Mapping[str, Any] | None,
        retain_optimizer_state: Mapping[str, Any] | None,
        initialize_kernel: bool,
    ) -> None:
        assert self.train_continuous is not None
        assert self.train_categorical is not None
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
        # Module constructors (notably nn.Embedding) only use Torch's global
        # generator.  Fork and seed the CPU generator so construction is driven
        # by this surrogate's dedicated stream without perturbing user code.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(module_seed)
            constraint = gpytorch.constraints.GreaterThan(self.config.noise_lower_bound)
            prior = gpytorch.priors.LogNormalPrior(math.log(self.config.noise_initial), 0.5)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=constraint, noise_prior=prior
            ).to(device=self.device, dtype=self.dtype)
            model = _MixedExactGP(
                self.train_continuous,
                self.train_categorical,
                self.train_targets,
                likelihood,
                category_sizes=self.category_sizes,
                ard=self.config.ard,
            ).to(device=self.device, dtype=self.dtype)
        likelihood.noise = max(self.config.noise_initial, self.config.noise_lower_bound)
        if initialize_kernel:
            initialize_numeric_lengthscales(
                model.covar_module,
                self.train_continuous,
                sample_limit=self.config.kernel_initialization_samples,
                lower_bound=self.config.lengthscale_lower_bound,
                generator=self.generator,
            )
            variance = self.train_targets.var(unbiased=False).clamp_min(torch.finfo(self.dtype).eps)
            model.covar_module.outputscale = variance
        if retain_model_state is not None:
            model.load_state_dict(retain_model_state, strict=False)
        if retain_likelihood_state is not None:
            likelihood.load_state_dict(retain_likelihood_state, strict=False)
        self.model = model
        self.likelihood = likelihood
        optimizer = self._create_optimizer()
        if retain_optimizer_state is not None:
            try:
                optimizer.load_state_dict(dict(retain_optimizer_state))
            except (KeyError, TypeError, ValueError, RuntimeError):
                if self.diagnostics is not None:
                    self.diagnostics.increment("gp.optimizer_state_incompatible")
        if isinstance(optimizer, PreconditionedSGLD):
            # A retained optimizer may come from a smaller training set.
            optimizer.set_observation_count(self.train_targets.numel())
        self.optimizer = optimizer

    def _create_optimizer(self) -> torch.optim.Optimizer:
        assert self.model is not None
        assert self.train_targets is not None
        return create_optimizer(
            self.config.optimizer,
            self.model.parameters(),
            learning_rate=self.config.learning_rate,
            observations=self.train_targets.numel(),
            pretrain_steps=max(1, self.config.initial_steps // 10),
            lbfgs_max_iter=self.config.lbfgs_max_iter,
            generator=self.generator,
        )

    def _settings(self, jitter: float) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            gpytorch.settings.cholesky_jitter(
                float_value=jitter, double_value=jitter, half_value=jitter
            )
        )
        if self.config.max_cholesky_size is not None:
            stack.enter_context(gpytorch.settings.max_cholesky_size(self.config.max_cholesky_size))
        if self.config.max_preconditioner_size is not None:
            stack.enter_context(
                gpytorch.settings.max_preconditioner_size(self.config.max_preconditioner_size)
            )
        if self.config.cg_tolerance is not None:
            stack.enter_context(gpytorch.settings.cg_tolerance(self.config.cg_tolerance))
        return stack

    def _optimize(self, kind: str, requested_steps: int) -> FitReport:
        assert self.model is not None
        assert self.likelihood is not None
        assert self.optimizer is not None
        assert self.train_continuous is not None
        assert self.train_categorical is not None
        assert self.train_targets is not None
        start = time.perf_counter()
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        completed = 0
        final_loss: float | None = None
        maximum_jitter = self.config.jitter_initial
        total_retries = 0
        early_stopped = False
        stable_steps = 0
        previous_loss: float | None = None
        failure: str | None = None

        try:
            for _ in range(requested_steps):
                jitter = self.config.jitter_initial
                step_retries = 0
                while True:
                    try:
                        final_loss = self._optimization_step(mll, jitter)
                        break
                    except RuntimeError as exc:
                        if not _looks_numerical(exc):
                            raise
                        step_retries += 1
                        total_retries += 1
                        jitter *= self.config.jitter_multiplier
                        maximum_jitter = max(maximum_jitter, jitter)
                        if (
                            step_retries > self.config.max_jitter_retries
                            or jitter > self.config.jitter_max
                        ):
                            raise NumericalError(
                                "exact-GP fitting failed after jitter escalation"
                            ) from exc
                completed += 1
                if not math.isfinite(final_loss):
                    raise NumericalError("exact-GP fitting produced a non-finite loss")
                if self.config.early_stopping and previous_loss is not None:
                    denominator = max(abs(previous_loss), torch.finfo(self.dtype).eps)
                    relative_change = abs(previous_loss - final_loss) / denominator
                    stable_steps = (
                        stable_steps + 1 if relative_change <= self.config.relative_tolerance else 0
                    )
                    if stable_steps >= self.config.patience:
                        early_stopped = True
                        break
                previous_loss = final_loss
        except (NumericalError, RuntimeError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            self.model.eval()
            self.likelihood.eval()
            report = FitReport(
                kind=kind,
                observations=self.train_targets.numel(),
                requested_steps=requested_steps,
                completed_steps=completed,
                final_loss=final_loss,
                wall_time=time.perf_counter() - start,
                maximum_jitter=maximum_jitter,
                jitter_retries=total_retries,
                early_stopped=early_stopped,
                failure=failure,
            )
            if self.diagnostics is not None:
                self.diagnostics.add_fit_report(report)
            raise NumericalError(failure) from exc

        self.model.eval()
        self.likelihood.eval()
        return FitReport(
            kind=kind,
            observations=self.train_targets.numel(),
            requested_steps=requested_steps,
            completed_steps=completed,
            final_loss=final_loss,
            wall_time=time.perf_counter() - start,
            maximum_jitter=maximum_jitter,
            jitter_retries=total_retries,
            early_stopped=early_stopped,
            failure=failure,
        )

    def _optimization_step(
        self, mll: gpytorch.mlls.ExactMarginalLogLikelihood, jitter: float
    ) -> float:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.train_continuous is not None
        assert self.train_categorical is not None
        assert self.train_targets is not None

        model = self.model
        optimizer = self.optimizer
        train_continuous = self.train_continuous
        train_categorical = self.train_categorical
        train_targets = self.train_targets

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            with self._settings(jitter):
                distribution = model(train_continuous, train_categorical)
                loss: torch.Tensor = -mll(distribution, train_targets)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        if isinstance(optimizer, torch.optim.LBFGS):
            loss = optimizer.step(closure)  # type: ignore[no-untyped-call]
            if loss is None:
                loss = closure().detach()
        else:
            loss = closure()
            optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def predict(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mean, variance, and scalar observation-noise variance."""

        if self.model is None or self.likelihood is None:
            raise RuntimeError("the GP has not been fitted")
        continuous, categorical = self._prepare_prediction_inputs(continuous, categorical)
        if self.model.training:
            self.model.eval()
        if self.likelihood.training:
            self.likelihood.eval()
        self.posterior_calls += 1
        if self.diagnostics is not None:
            self.diagnostics.increment("posterior.calls")
            self.diagnostics.increment("posterior.candidates", continuous.shape[0])
        jitter = self.config.jitter_initial
        retries = 0
        while True:
            try:
                with (
                    self._settings(jitter),
                    gpytorch.settings.fast_pred_var(self.config.fast_pred_var),
                ):
                    if self.config.eval_cg_tolerance is None:
                        distribution = self.model(continuous, categorical)
                    else:
                        with gpytorch.settings.eval_cg_tolerance(self.config.eval_cg_tolerance):
                            distribution = self.model(continuous, categorical)
                    if self.config.predict_observation_noise:
                        distribution = self.likelihood(distribution)
                mean = distribution.mean.reshape(-1)
                variance = distribution.variance.reshape(-1).clamp_min(torch.finfo(self.dtype).eps)
                if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
                    raise NumericalError("exact-GP posterior contains non-finite values")
                return mean, variance, self.noise_variance
            except RuntimeError as exc:
                if not _looks_numerical(exc):
                    raise
                retries += 1
                jitter *= self.config.jitter_multiplier
                if retries > self.config.max_jitter_retries or jitter > self.config.jitter_max:
                    raise NumericalError(
                        "exact-GP posterior failed after jitter escalation"
                    ) from exc

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "input_scaler_kind": (
                "minmax" if isinstance(self.input_scaler, TorchMinMaxScaler) else "identity"
            ),
            "input_scaler": self.input_scaler.state_dict(),
            "model": None if self.model is None else self.model.state_dict(),
            "likelihood": (None if self.likelihood is None else self.likelihood.state_dict()),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "train_continuous": self.train_continuous,
            "train_categorical": self.train_categorical,
            "train_targets": self.train_targets,
            "transform_version": self.transform_version,
            "fit_count": self.fit_count,
            "update_count": self.update_count,
            "full_refit_count": self.full_refit_count,
            "updates_since_full_refit": self.updates_since_full_refit,
            "last_full_fit_observations": self.last_full_fit_observations,
            "posterior_calls": self.posterior_calls,
            "parameter_version": self.parameter_version,
            "posterior_cache_version": self.posterior_cache_version,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != _STATE_SCHEMA_VERSION:
            raise ValueError("unsupported exact-GP state schema")
        expected_scaler_kind = "minmax" if self.num_continuous else "identity"
        if state.get("input_scaler_kind") != expected_scaler_kind:
            raise ValueError("exact-GP input scaler is incompatible with the model dimensions")
        scaler_state = state.get("input_scaler")
        if not isinstance(scaler_state, Mapping):
            raise ValueError("exact-GP input scaler state is missing or malformed")
        self.input_scaler.load_state_dict(dict(scaler_state))
        self.input_scaler.to(device=self.device, dtype=self.dtype)
        train_continuous = state.get("train_continuous")
        train_categorical = state.get("train_categorical")
        train_targets = state.get("train_targets")
        if train_continuous is not None:
            assert train_categorical is not None and train_targets is not None
            self.train_continuous = torch.as_tensor(
                train_continuous, device=self.device, dtype=self.dtype
            )
            self.train_categorical = torch.as_tensor(
                train_categorical, device=self.device, dtype=torch.int64
            )
            self.train_targets = torch.as_tensor(
                train_targets, device=self.device, dtype=self.dtype
            )
            self._construct_model(
                retain_model_state=state.get("model"),
                retain_likelihood_state=state.get("likelihood"),
                retain_optimizer_state=state.get("optimizer"),
                initialize_kernel=False,
            )
            assert self.model is not None and self.likelihood is not None
            self.model.eval()
            self.likelihood.eval()
        self.transform_version = int(state["transform_version"])
        self.fit_count = int(state["fit_count"])
        self.update_count = int(state["update_count"])
        self.full_refit_count = int(state["full_refit_count"])
        self.updates_since_full_refit = int(state.get("updates_since_full_refit", 0))
        self.last_full_fit_observations = int(state["last_full_fit_observations"])
        self.posterior_calls = int(state["posterior_calls"])
        self.parameter_version = int(state["parameter_version"])
        self.posterior_cache_version = int(state["posterior_cache_version"])


def _looks_numerical(error: RuntimeError) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "cholesky",
            "not positive definite",
            "singular",
            "nan",
            "inf",
            "symeig",
            "linalg",
        )
    )
