# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Explicit, serializable configuration for LeanHEBO."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeVar


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Device, precision, randomness, and evaluation controls."""

    device: str = "cpu"
    dtype: Literal["float32", "float64"] = "float32"
    seed: int | None = None
    deterministic: bool = False
    acquisition_batch_size: int | None = 4096
    synchronize_device_for_timing: bool = True
    enable_torch_compile: bool = False

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("device must be a non-empty Torch device string")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.acquisition_batch_size is not None and self.acquisition_batch_size < 1:
            raise ValueError("acquisition_batch_size must be positive or None")


@dataclass(frozen=True, slots=True)
class GPConfig:
    """Exact-GP fitting and lifecycle controls."""

    learning_rate: float = 1e-2
    optimizer: Literal["psgld", "adam", "lbfgs"] = "psgld"
    initial_steps: int = 100
    update_steps: int = 10
    full_refit_interval: int | None = 25
    full_refit_growth_factor: float | None = 1.5
    reuse_parameters: bool = True
    reuse_optimizer_state: bool = True
    use_set_train_data: bool = True
    use_fantasy_updates: bool = False
    noise_lower_bound: float = 8e-4
    noise_initial: float = 1e-2
    predict_observation_noise: bool = False
    ard: bool = True
    early_stopping: bool = False
    patience: int = 10
    relative_tolerance: float = 1e-4
    max_cholesky_size: int | None = None
    max_preconditioner_size: int | None = None
    cg_tolerance: float | None = None
    eval_cg_tolerance: float | None = None
    fast_pred_var: bool = True
    kernel_initialization_samples: int = 1000
    lengthscale_lower_bound: float = 0.02
    jitter_initial: float = 1e-8
    jitter_multiplier: float = 10.0
    jitter_max: float = 1.0
    max_jitter_retries: int = 9
    lbfgs_max_iter: int = 5

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.initial_steps < 0 or self.update_steps < 0:
            raise ValueError("GP step counts cannot be negative")
        if self.full_refit_interval is not None and self.full_refit_interval < 1:
            raise ValueError("full_refit_interval must be positive or None")
        if self.full_refit_growth_factor is not None and (
            not math.isfinite(self.full_refit_growth_factor) or self.full_refit_growth_factor <= 1
        ):
            raise ValueError("full_refit_growth_factor must exceed 1 or be None")
        if (
            not math.isfinite(self.noise_lower_bound)
            or not math.isfinite(self.noise_initial)
            or self.noise_lower_bound <= 0
            or self.noise_initial <= 0
        ):
            raise ValueError("noise bounds and initial value must be positive and finite")
        if self.noise_initial <= self.noise_lower_bound:
            raise ValueError("noise_initial must be strictly greater than noise_lower_bound")
        if self.use_fantasy_updates:
            raise ValueError(
                "use_fantasy_updates=True is not implemented; use the persistent "
                "set_train_data update path"
            )
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be finite and non-negative")
        if self.kernel_initialization_samples < 2:
            raise ValueError("kernel_initialization_samples must be at least 2")
        if not math.isfinite(self.lengthscale_lower_bound) or self.lengthscale_lower_bound <= 0:
            raise ValueError("lengthscale_lower_bound must be positive and finite")
        if (
            not math.isfinite(self.jitter_initial)
            or not math.isfinite(self.jitter_multiplier)
            or self.jitter_initial <= 0
            or self.jitter_multiplier <= 1
        ):
            raise ValueError("invalid jitter schedule")
        if (
            not math.isfinite(self.jitter_max)
            or self.jitter_max < self.jitter_initial
            or self.max_jitter_retries < 0
        ):
            raise ValueError("invalid maximum jitter settings")
        if self.lbfgs_max_iter < 1:
            raise ValueError("lbfgs_max_iter must be positive")
        if self.max_cholesky_size is not None and self.max_cholesky_size < 0:
            raise ValueError("max_cholesky_size cannot be negative")
        if self.max_preconditioner_size is not None and self.max_preconditioner_size < 0:
            raise ValueError("max_preconditioner_size cannot be negative")
        for name, value in (
            ("cg_tolerance", self.cg_tolerance),
            ("eval_cg_tolerance", self.eval_cg_tolerance),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite or None")


@dataclass(frozen=True, slots=True)
class WarpConfig:
    """Output standardization and power-transform controls."""

    method: Literal["auto", "none", "box-cox", "yeo-johnson"] = "auto"
    standardize_before_warp: bool = True
    refit_interval: int | None = 1
    minimum_points: int = 3
    minimum_transformed_std: float = 0.5
    lambda_lower_bound: float = -5.0
    lambda_upper_bound: float = 5.0
    lambda_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.refit_interval is not None and self.refit_interval < 1:
            raise ValueError("refit_interval must be positive or None")
        if self.minimum_points < 1:
            raise ValueError("minimum_points must be positive")
        if not math.isfinite(self.minimum_transformed_std) or self.minimum_transformed_std < 0:
            raise ValueError("minimum_transformed_std must be finite and non-negative")
        if (
            not math.isfinite(self.lambda_lower_bound)
            or not math.isfinite(self.lambda_upper_bound)
            or self.lambda_lower_bound >= self.lambda_upper_bound
        ):
            raise ValueError("lambda bounds must be strictly increasing")
        if not math.isfinite(self.lambda_tolerance) or self.lambda_tolerance <= 0:
            raise ValueError("lambda_tolerance must be positive and finite")


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    """MACE policy controls."""

    epsilon: float = 1e-4
    upsi: float = 0.5
    delta: float = 0.01
    kappa: float | None = None
    stochastic: bool = True
    posterior_cache: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")
        if not math.isfinite(self.upsi) or self.upsi <= 0:
            raise ValueError("upsi must be positive and finite")
        if not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if self.kappa is not None and (not math.isfinite(self.kappa) or self.kappa < 0):
            raise ValueError("kappa must be finite and non-negative or None")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Tensor-native NSGA-II controls."""

    population_size: int = 100
    generations: int = 100
    crossover_probability: float = 0.9
    crossover_eta: float = 15.0
    mutation_probability: float | None = None
    mutation_eta: float = 20.0
    tournament_size: int = 2
    eliminate_duplicates: bool = True
    reuse_previous_population: bool = False
    keep_history: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 0:
            raise ValueError("generations cannot be negative")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability must be between zero and one")
        if (
            not math.isfinite(self.crossover_eta)
            or not math.isfinite(self.mutation_eta)
            or self.crossover_eta <= 0
            or self.mutation_eta <= 0
        ):
            raise ValueError("crossover_eta and mutation_eta must be positive and finite")
        if self.mutation_probability is not None and not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be between zero and one")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")


@dataclass(frozen=True, slots=True)
class LeanHEBOConfig:
    """Complete LeanHEBO configuration; there are deliberately no presets."""

    random_samples: int | None = None
    nonfinite_policy: Literal["drop", "raise"] = "drop"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.random_samples is not None and self.random_samples < 2:
            raise ValueError("random_samples must be at least 2 or None")
        if self.nonfinite_policy not in ("drop", "raise"):
            raise ValueError("nonfinite_policy must be 'drop' or 'raise'")

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint- and JSON-friendly representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LeanHEBOConfig:
        """Reconstruct a configuration from :meth:`to_dict` output."""

        root = dict(value)
        return cls(
            random_samples=root.get("random_samples"),
            nonfinite_policy=root.get("nonfinite_policy", "drop"),
            runtime=_construct(RuntimeConfig, root.get("runtime", {})),
            gp=_construct(GPConfig, root.get("gp", {})),
            warp=_construct(WarpConfig, root.get("warp", {})),
            acquisition=_construct(AcquisitionConfig, root.get("acquisition", {})),
            search=_construct(SearchConfig, root.get("search", {})),
        )


ConfigT = TypeVar("ConfigT", RuntimeConfig, GPConfig, WarpConfig, AcquisitionConfig, SearchConfig)


def _construct(config_type: type[ConfigT], value: object) -> ConfigT:
    if isinstance(value, config_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping for {config_type.__name__}")
    return config_type(**dict(value))
