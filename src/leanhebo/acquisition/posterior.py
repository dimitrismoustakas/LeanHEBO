# SPDX-License-Identifier: MIT

"""One-call-per-chunk exact-GP posterior evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class PosteriorProvider(Protocol):
    posterior_cache_version: int

    def predict(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class PosteriorStats:
    mean: torch.Tensor
    variance: torch.Tensor
    stddev: torch.Tensor
    noise_variance: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> PosteriorStats:
        return PosteriorStats(
            mean=self.mean.index_select(0, indices),
            variance=self.variance.index_select(0, indices),
            stddev=self.stddev.index_select(0, indices),
            noise_variance=self.noise_variance,
        )


class PosteriorEvaluator:
    """Evaluate and optionally cache shared posterior statistics."""

    def __init__(
        self,
        provider: PosteriorProvider,
        *,
        batch_size: int | None = 4096,
        cache: bool = True,
    ) -> None:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")
        self.provider = provider
        self.batch_size = batch_size
        self.cache = cache
        self._cache_key: tuple[object, ...] | None = None
        self._cache_value: PosteriorStats | None = None
        self._cache_inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def invalidate(self) -> None:
        self._cache_key = None
        self._cache_value = None
        self._cache_inputs = None

    def evaluate(self, continuous: torch.Tensor, categorical: torch.Tensor) -> PosteriorStats:
        if continuous.shape[0] != categorical.shape[0]:
            raise ValueError("continuous and categorical batch lengths differ")
        key = self._key(continuous, categorical)
        if self.cache and key == self._cache_key and self._cache_value is not None:
            return self._cache_value
        count = continuous.shape[0]
        if count == 0:
            empty = continuous.new_empty((0,))
            result = PosteriorStats(empty, empty, empty, continuous.new_zeros(()))
            if self.cache:
                self._cache_key = key
                self._cache_value = result
                self._cache_inputs = (continuous, categorical)
            return result
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
        result = PosteriorStats(mean, variance, variance.sqrt(), noise)
        if self.cache:
            self._cache_key = key
            self._cache_value = result
            # Retaining the input objects prevents allocator pointer reuse from
            # making unrelated tensors look like the cached evaluation.
            self._cache_inputs = (continuous, categorical)
        return result

    def _key(self, continuous: torch.Tensor, categorical: torch.Tensor) -> tuple[object, ...]:
        return (
            self.provider.posterior_cache_version,
            id(continuous),
            continuous.data_ptr(),
            tuple(continuous.shape),
            tuple(continuous.stride()),
            continuous.storage_offset(),
            continuous.dtype,
            continuous.device,
            continuous._version,
            id(categorical),
            categorical.data_ptr(),
            tuple(categorical.shape),
            tuple(categorical.stride()),
            categorical.storage_offset(),
            categorical.dtype,
            categorical.device,
            categorical._version,
        )
