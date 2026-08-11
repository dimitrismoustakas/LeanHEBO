# SPDX-License-Identifier: MIT

"""One-call-per-chunk exact-GP posterior evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class PosteriorProvider(Protocol):
    def predict(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class PosteriorStats:
    mean: torch.Tensor
    variance: torch.Tensor
    stddev: torch.Tensor
    noise_variance: torch.Tensor


class PosteriorEvaluator:
    """Evaluate shared posterior statistics in bounded chunks."""

    def __init__(
        self,
        provider: PosteriorProvider,
        *,
        batch_size: int | None = 4096,
    ) -> None:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")
        self.provider = provider
        self.batch_size = batch_size

    def evaluate(self, continuous: torch.Tensor, categorical: torch.Tensor) -> PosteriorStats:
        if continuous.shape[0] != categorical.shape[0]:
            raise ValueError("continuous and categorical batch lengths differ")
        count = continuous.shape[0]
        if count == 0:
            empty = continuous.new_empty((0,))
            return PosteriorStats(empty, empty, empty, continuous.new_zeros(()))
        chunk_size = count if self.batch_size is None else self.batch_size
        means: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        noise: torch.Tensor | None = None
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            mean, variance, chunk_noise = self.provider.predict(
                continuous[start:end], categorical[start:end]
            )
            means.append(mean)
            variances.append(variance)
            noise = chunk_noise if noise is None else noise
        mean = torch.cat(means)
        variance = torch.cat(variances).clamp_min(torch.finfo(continuous.dtype).eps)
        assert noise is not None
        return PosteriorStats(mean, variance, variance.sqrt(), noise)
