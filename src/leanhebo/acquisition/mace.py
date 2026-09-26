# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""HEBO's three-objective MACE acquisition from shared posterior statistics."""

from __future__ import annotations

import math

import torch

from leanhebo.acquisition.posterior import PosteriorEvaluator, PosteriorStats
from leanhebo.errors import NumericalError


def _log_expected_improvement(normalized: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
    """Evaluate log(sigma * (phi(z) + z * Phi(z))) without tail cancellation."""

    upper = normalized.clamp_min(-1.0)
    upper_density = torch.exp(-0.5 * upper.square()) / math.sqrt(2.0 * math.pi)
    upper_log_ei = torch.log(upper_density + upper * torch.special.ndtr(upper))

    # For z <= -1, factor out phi(z), then evaluate log(1 - |z| Phi(z) / phi(z)).
    # erfcx removes the Gaussian exponential; expm1 retains the small difference.
    # This is the stable LogEI identity also used by BoTorch's analytic LogEI.
    magnitude = (-normalized).clamp_min(1.0)
    tail_bound = 1e6 if normalized.dtype == torch.float64 else 1e3
    bounded = magnitude.clamp_max(tail_bound)
    log_ratio = torch.log(bounded * torch.special.erfcx(bounded / math.sqrt(2.0))) + 0.5 * math.log(
        math.pi / 2.0
    )
    log_remainder = torch.where(
        magnitude < tail_bound,
        torch.log(-torch.expm1(log_ratio)),
        # Beyond this bound, the next asymptotic term is below log-EI precision.
        -2.0 * torch.log(magnitude),
    )
    lower_log_ei = -0.5 * magnitude.square() - 0.5 * math.log(2.0 * math.pi) + log_remainder
    return torch.log(stddev) + torch.where(normalized > -1.0, upper_log_ei, lower_log_ei)


class MACEEvaluator:
    """Compute stochastic LCB, negative log-EI, and negative log-PI."""

    def __init__(
        self,
        posterior: PosteriorEvaluator,
        *,
        best_y: torch.Tensor | float,
        kappa: float,
        epsilon: float = 1e-4,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
        validate: bool = True,
    ) -> None:
        if kappa < 0:
            raise ValueError("kappa cannot be negative")
        if epsilon < 0:
            raise ValueError("epsilon cannot be negative")
        self.posterior = posterior
        self.best_y = best_y
        self.kappa = kappa
        self.epsilon = epsilon
        self.stochastic = stochastic
        self.generator = generator
        self.validate = validate

    def evaluate(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        return self.from_stats(self.posterior.evaluate(continuous, categorical))

    def from_stats(self, stats: PosteriorStats) -> torch.Tensor:
        mean = stats.mean
        stddev = stats.stddev.clamp_min(torch.finfo(stats.stddev.dtype).eps)
        tau = torch.as_tensor(self.best_y, device=mean.device, dtype=mean.dtype)
        if self.stochastic:
            noise_stddev = (2.0 * stats.noise_variance).sqrt()
            lcb_noise = torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=self.generator,
            )
            improvement_noise = torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=self.generator,
            )
            noisy_lcb_mean = mean + noise_stddev * lcb_noise
            improvement_mean = mean + noise_stddev * improvement_noise
        else:
            noisy_lcb_mean = mean
            improvement_mean = mean
        lcb = noisy_lcb_mean - self.kappa * stddev
        normalized = (tau - self.epsilon - improvement_mean) / stddev

        negative_log_ei = -_log_expected_improvement(normalized, stddev)
        negative_log_pi = -torch.special.log_ndtr(normalized)
        objectives = torch.stack((lcb, negative_log_ei, negative_log_pi), dim=-1)
        if self.validate and not torch.isfinite(objectives).all():
            bad = int((~torch.isfinite(objectives)).sum().item())
            raise NumericalError(f"MACE produced {bad} non-finite objective values")
        return objectives

    __call__ = evaluate
