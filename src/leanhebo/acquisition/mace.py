# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""HEBO's three-objective MACE acquisition from shared posterior statistics."""

from __future__ import annotations

import math

import torch

from leanhebo.acquisition.posterior import PosteriorEvaluator, PosteriorStats
from leanhebo.errors import NumericalError


class MACEEvaluator:
    """Compute stochastic LCB, negative log-EI, and negative log-PI."""

    num_objectives = 3

    def __init__(
        self,
        posterior: PosteriorEvaluator,
        *,
        best_y: torch.Tensor | float,
        kappa: float,
        epsilon: float = 1e-4,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
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

        log_phi = -0.5 * normalized.square() - 0.5 * math.log(2.0 * math.pi)
        probability = torch.special.ndtr(normalized)
        expected_improvement = stddev * (probability * normalized + torch.exp(log_phi))
        log_ei = torch.log(expected_improvement)
        log_pi = torch.log(probability)
        log_ei_approx = (
            torch.log(stddev) - 0.5 * normalized.square() - torch.log(normalized.square() - 1.0)
        )
        log_pi_approx = (
            -0.5 * normalized.square() - torch.log(-normalized) - 0.5 * math.log(2.0 * math.pi)
        )
        direct = (normalized > -6.0) & torch.isfinite(log_ei) & torch.isfinite(log_pi)
        negative_log_ei = -torch.where(direct, log_ei, log_ei_approx)
        negative_log_pi = -torch.where(direct, log_pi, log_pi_approx)
        objectives = torch.stack((lcb, negative_log_ei, negative_log_pi), dim=-1)
        if not torch.isfinite(objectives).all():
            bad = int((~torch.isfinite(objectives)).sum().item())
            raise NumericalError(f"MACE produced {bad} non-finite objective values")
        return objectives

    __call__ = evaluate
