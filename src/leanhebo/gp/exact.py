# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Persistent HEBO-compatible exact Gaussian-process surrogate."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any

import gpytorch  # type: ignore[import-untyped]
import torch

from leanhebo.config import GPConfig, RuntimeConfig
from leanhebo.diagnostics import Diagnostics, FitReport
from leanhebo.errors import NumericalError
from leanhebo.gp.kernel import (
    MixedFeatureExtractor,
    build_kernel,
    initialize_base_numeric_lengthscales,
)
from leanhebo.transforms import IdentityScaler, TorchMinMaxScaler


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
    """Own an exact GP, likelihood, optimizer, training state, and prediction lifecycle.

    Subclasses adapt the lifecycle through narrow hooks (`_store_training_data`,
    `_train_inputs`, `_build_model`, `_initialize_kernel`, `_prepare_prediction_inputs`,
    and `_restore_extra_training_state`) instead of overriding `fit` or `predict`.
    """

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
        self._category_sizes_tensor = torch.tensor(
            self.category_sizes,
            device=self.device,
            dtype=torch.int64,
        )
        self.generator = generator
        self.diagnostics = diagnostics
        self.input_scaler: IdentityScaler | TorchMinMaxScaler
        if self.num_continuous:
            self.input_scaler = TorchMinMaxScaler(feature_range=(-1.0, 1.0))
        else:
            self.input_scaler = IdentityScaler()
        self.input_scaler.to(device=self.device, dtype=self.dtype)
        self.model: gpytorch.models.ExactGP | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.optimizer: torch.optim.Adam | None = None
        self.train_continuous: torch.Tensor | None = None
        self.train_categorical: torch.Tensor | None = None
        self.train_targets: torch.Tensor | None = None
        self.transform_version = -1
        self.updates_since_full_refit = 0
        self.last_full_fit_observations = 0

    @property
    def fitted(self) -> bool:
        return self.model is not None

    @property
    def noise_variance(self) -> torch.Tensor:
        if self.likelihood is None:
            raise RuntimeError("the GP has not been fitted")
        noise: torch.Tensor = self.likelihood.noise.detach().reshape(())
        return noise

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
        if bool(((categorical < 0) | (categorical >= self._category_sizes_tensor)).any()):
            raise ValueError("categorical codes are out of bounds")
        return continuous.contiguous(), categorical.contiguous()

    def _prepare_prediction_inputs(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        continuous, categorical = self._coerce_inputs(continuous, categorical)
        if self.num_continuous and not self.input_scaler.fitted:
            raise RuntimeError("the GP input scaler has not been fitted")
        return self.input_scaler.transform(continuous).contiguous(), categorical

    def _store_training_data(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Refit input scaling and retain the exact tensors the model will train on."""

        self.input_scaler.fit(continuous)
        self.train_continuous = self.input_scaler.transform(continuous).contiguous()
        self.train_categorical = categorical
        self.train_targets = targets

    def _train_inputs(self) -> tuple[torch.Tensor, ...]:
        assert self.train_continuous is not None
        assert self.train_categorical is not None
        return (self.train_continuous, self.train_categorical)

    def fit(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
        *,
        transform_version: int,
        force_full_refit: bool = False,
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
        full_refit = first_fit or force_full_refit or self._full_refit_due(targets.numel())
        kind = "initial" if first_fit else ("full_refit" if full_refit else "update")
        steps = self.config.initial_steps if full_refit else self.config.update_steps
        if self.transform_version != transform_version and self.diagnostics is not None:
            self.diagnostics.increment("gp.transform_invalidations")

        self._store_training_data(continuous, categorical, targets)
        self.transform_version = transform_version

        if full_refit or not self.config.use_set_train_data or not self.config.reuse_parameters:
            retain_model_state: Mapping[str, Any] | None = None
            retain_likelihood_state: Mapping[str, Any] | None = None
            if self.config.reuse_parameters and self.model is not None:
                retain_model_state = self.model.state_dict()
                assert self.likelihood is not None
                retain_likelihood_state = self.likelihood.state_dict()
            self._construct_model(
                retain_model_state=retain_model_state,
                retain_likelihood_state=retain_likelihood_state,
                retain_optimizer_state=(
                    self.optimizer.state_dict()
                    if not full_refit
                    and steps > 0
                    and self.config.reuse_optimizer_state
                    and self.optimizer is not None
                    else None
                ),
                initialize_kernel=first_fit or not self.config.reuse_parameters,
            )
        else:
            assert self.model is not None
            self.model.set_train_data(inputs=self._train_inputs(), targets=targets, strict=False)
            if not self.config.reuse_optimizer_state:
                self.optimizer = self._create_optimizer()

        report = self._optimize(kind, steps)
        if full_refit:
            self.updates_since_full_refit = 0
            self.last_full_fit_observations = targets.numel()
        else:
            self.updates_since_full_refit += 1
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

    def _build_model(
        self, likelihood: gpytorch.likelihoods.GaussianLikelihood
    ) -> gpytorch.models.ExactGP:
        assert self.train_continuous is not None
        assert self.train_categorical is not None
        assert self.train_targets is not None
        return _MixedExactGP(
            self.train_continuous,
            self.train_categorical,
            self.train_targets,
            likelihood,
            category_sizes=self.category_sizes,
            ard=self.config.ard,
        )

    def _initialize_kernel(self, model: gpytorch.models.ExactGP) -> None:
        assert self.train_continuous is not None
        initialize_base_numeric_lengthscales(
            model.covar_module.base_kernel,
            self.train_continuous,
            sample_limit=self.config.kernel_initialization_samples,
            lower_bound=self.config.lengthscale_lower_bound,
            generator=self.generator,
        )

    def _construct_model(
        self,
        *,
        retain_model_state: Mapping[str, Any] | None,
        retain_likelihood_state: Mapping[str, Any] | None,
        retain_optimizer_state: Mapping[str, Any] | None,
        initialize_kernel: bool,
    ) -> None:
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
            model = self._build_model(likelihood).to(device=self.device, dtype=self.dtype)
        likelihood.noise = max(self.config.noise_initial, self.config.noise_lower_bound)
        if initialize_kernel:
            self._initialize_kernel(model)
            variance = self.train_targets.var(unbiased=False).clamp_min(torch.finfo(self.dtype).eps)
            model.covar_module.outputscale = variance
        if retain_model_state is not None:
            model.load_state_dict(retain_model_state)
        if retain_likelihood_state is not None:
            likelihood.load_state_dict(retain_likelihood_state)
        self.model = model
        self.likelihood = likelihood
        self.optimizer = self._create_optimizer()
        if retain_optimizer_state is not None:
            self.optimizer.load_state_dict(dict(retain_optimizer_state))

    def _create_optimizer(self) -> torch.optim.Adam:
        assert self.model is not None
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.99),
        )

    def _settings(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            gpytorch.settings.cholesky_jitter(
                float_value=self.config.jitter,
                double_value=self.config.jitter,
                half_value=self.config.jitter,
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
        assert self.train_targets is not None
        start = time.perf_counter()
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        completed = 0
        final_loss: float | None = None
        early_stopped = False
        stable_steps = 0
        previous_loss: float | None = None
        failure: str | None = None

        try:
            for _ in range(requested_steps):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._training_loss(mll)
                if self.config.early_stopping or loss.device.type != "cuda":
                    current_loss = float(loss.detach().cpu())
                    final_loss = current_loss
                    if not math.isfinite(current_loss):
                        raise NumericalError("exact-GP fitting produced a non-finite loss")
                    if self.config.early_stopping and previous_loss is not None:
                        denominator = max(abs(previous_loss), torch.finfo(self.dtype).eps)
                        relative_change = abs(previous_loss - current_loss) / denominator
                        stable_steps = (
                            stable_steps + 1
                            if relative_change <= self.config.relative_tolerance
                            else 0
                        )
                        if stable_steps >= self.config.patience:
                            early_stopped = True
                            break
                    previous_loss = current_loss
                loss.backward()  # type: ignore[no-untyped-call]
                final_loss = None
                self.optimizer.step()
                completed += 1
            if completed and not early_stopped:
                final_loss = None
                with torch.no_grad():
                    loss = self._training_loss(mll)
                final_loss = float(loss.detach().cpu())
                if not math.isfinite(final_loss):
                    raise NumericalError("exact-GP fitting produced a non-finite loss")
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
            early_stopped=early_stopped,
            failure=failure,
        )

    def _training_loss(self, mll: gpytorch.mlls.ExactMarginalLogLikelihood) -> torch.Tensor:
        assert self.model is not None
        assert self.train_targets is not None

        with self._settings():
            distribution = self.model(*self._train_inputs())
            loss: torch.Tensor = -mll(distribution, self.train_targets)
        return loss

    @torch.no_grad()
    def predict(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mean, variance, and scalar observation-noise variance."""

        if self.model is None or self.likelihood is None:
            raise RuntimeError("the GP has not been fitted")
        inputs = self._prepare_prediction_inputs(continuous, categorical)
        if self.model.training:
            self.model.eval()
        if self.likelihood.training:
            self.likelihood.eval()
        if self.diagnostics is not None:
            self.diagnostics.increment("posterior.calls")
            self.diagnostics.increment("posterior.candidates", inputs[0].shape[0])
        with (
            self._settings(),
            gpytorch.settings.fast_pred_var(self.config.fast_pred_var),
        ):
            if self.config.eval_cg_tolerance is None:
                distribution = self.model(*inputs)
            else:
                with gpytorch.settings.eval_cg_tolerance(self.config.eval_cg_tolerance):
                    distribution = self.model(*inputs)
            if self.config.predict_observation_noise:
                distribution = self.likelihood(distribution)
        mean = distribution.mean.reshape(-1)
        variance = distribution.variance.reshape(-1).clamp_min(torch.finfo(self.dtype).eps)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise NumericalError("exact-GP posterior contains non-finite values")
        return mean, variance, self.noise_variance

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_scaler": self.input_scaler.state_dict(),
            "model": None if self.model is None else self.model.state_dict(),
            "likelihood": (None if self.likelihood is None else self.likelihood.state_dict()),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "train_continuous": self.train_continuous,
            "train_categorical": self.train_categorical,
            "train_targets": self.train_targets,
            "transform_version": self.transform_version,
            "updates_since_full_refit": self.updates_since_full_refit,
            "last_full_fit_observations": self.last_full_fit_observations,
        }

    def _restore_extra_training_state(self, state: Mapping[str, Any]) -> None:
        """Restore subclass training tensors that model reconstruction depends on."""

        del state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        scaler_state = state["input_scaler"]
        if not isinstance(scaler_state, Mapping):
            raise ValueError("exact-GP input scaler state is missing or malformed")
        self.input_scaler.load_state_dict(dict(scaler_state))
        self.input_scaler.to(device=self.device, dtype=self.dtype)
        train_continuous = state["train_continuous"]
        train_categorical = state["train_categorical"]
        train_targets = state["train_targets"]
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
            self._restore_extra_training_state(state)
            self._construct_model(
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
