# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""HEBO-compatible mixed numeric/categorical feature and kernel construction."""

from __future__ import annotations

from collections.abc import Sequence

import gpytorch  # type: ignore[import-untyped]
import torch
from torch import nn

_PAIRWISE_DISTANCE_ELEMENT_BUDGET = 8_000_000


class MixedFeatureExtractor(nn.Module):
    """Concatenate numeric inputs and compact learned categorical embeddings."""

    def __init__(self, num_continuous: int, category_sizes: Sequence[int]) -> None:
        super().__init__()
        self.num_continuous = num_continuous
        self.category_sizes = tuple(int(size) for size in category_sizes)
        if any(size < 1 for size in self.category_sizes):
            raise ValueError("every categorical dimension must contain at least one value")
        embedding_sizes = tuple(min(50, 1 + size // 2) for size in self.category_sizes)
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_size)
            for size, embedding_size in zip(self.category_sizes, embedding_sizes, strict=True)
        )
        self.embedding_sizes = embedding_sizes
        self.output_dimensions = num_continuous + sum(embedding_sizes)

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if continuous.ndim != 2 or categorical.ndim != 2:
            raise ValueError("GP inputs must be rank-two tensors")
        if continuous.shape[0] != categorical.shape[0]:
            raise ValueError("continuous and categorical batch lengths differ")
        if continuous.shape[1] != self.num_continuous:
            raise ValueError("unexpected number of continuous GP inputs")
        if categorical.shape[1] != len(self.embeddings):
            raise ValueError("unexpected number of categorical GP inputs")
        if not self.embeddings:
            return continuous
        embedded = [
            embedding(categorical[:, index]).reshape(categorical.shape[0], -1)
            for index, embedding in enumerate(self.embeddings)
        ]
        return torch.cat([continuous, *embedded], dim=-1)


def build_kernel(
    *,
    num_continuous: int,
    feature_extractor: MixedFeatureExtractor,
    ard: bool,
) -> gpytorch.kernels.ScaleKernel:
    """Construct the product Matérn-3/2 covariance used by the main HEBO GP path."""

    return gpytorch.kernels.ScaleKernel(
        build_base_kernel(
            num_continuous=num_continuous,
            feature_extractor=feature_extractor,
            ard=ard,
        ),
        outputscale_prior=gpytorch.priors.GammaPrior(0.5, 0.5),
    )


def build_base_kernel(
    *,
    num_continuous: int,
    feature_extractor: MixedFeatureExtractor,
    ard: bool,
) -> gpytorch.kernels.Kernel:
    """Construct the unit-diagonal mixed Matérn base without an output scale."""

    components: list[gpytorch.kernels.Kernel] = []
    if num_continuous:
        components.append(
            gpytorch.kernels.MaternKernel(
                nu=1.5,
                ard_num_dims=num_continuous if ard else None,
                active_dims=torch.arange(num_continuous),
            )
        )
    if feature_extractor.embedding_sizes:
        components.append(
            gpytorch.kernels.MaternKernel(
                nu=1.5,
                active_dims=torch.arange(num_continuous, feature_extractor.output_dimensions),
            )
        )
    if not components:
        raise ValueError("an exact GP requires at least one non-fixed input dimension")
    return components[0] if len(components) == 1 else gpytorch.kernels.ProductKernel(*components)


def initialize_numeric_lengthscales(
    kernel: gpytorch.kernels.ScaleKernel,
    continuous: torch.Tensor,
    *,
    sample_limit: int,
    lower_bound: float,
    generator: torch.Generator,
) -> None:
    """Set ARD lengthscales from capped exact per-dimension pairwise distances."""

    if continuous.shape[1] == 0 or not isinstance(kernel.base_kernel, gpytorch.kernels.Kernel):
        return
    initialize_base_numeric_lengthscales(
        kernel.base_kernel,
        continuous,
        sample_limit=sample_limit,
        lower_bound=lower_bound,
        generator=generator,
    )


def initialize_base_numeric_lengthscales(
    base: gpytorch.kernels.Kernel,
    continuous: torch.Tensor,
    *,
    sample_limit: int,
    lower_bound: float,
    generator: torch.Generator,
) -> None:
    """Initialize an unscaled mixed base from numeric pairwise distances."""

    if continuous.shape[1] == 0:
        return
    numeric: gpytorch.kernels.MaternKernel | None = None
    if isinstance(base, gpytorch.kernels.MaternKernel):
        if base.ard_num_dims is not None:
            numeric = base
    elif isinstance(base, gpytorch.kernels.ProductKernel):
        for component in base.kernels:
            if isinstance(component, gpytorch.kernels.MaternKernel) and component.ard_num_dims:
                numeric = component
                break
    if numeric is None:
        return
    count = min(continuous.shape[0], sample_limit)
    if count < 2:
        return
    indices = torch.randperm(continuous.shape[0], device=continuous.device, generator=generator)[
        :count
    ]
    sample = continuous.index_select(0, indices)
    # HEBO initializes each ARD coordinate from median pairwise distance, not
    # median adjacent spacing. Reuse one pair index and batch dimensions under a
    # fixed element budget to preserve the exact statistic without O(n² d) memory.
    pairs = torch.triu_indices(count, count, offset=1, device=sample.device)
    pair_count = pairs.shape[1]
    columns_per_batch = max(1, _PAIRWISE_DISTANCE_ELEMENT_BUDGET // max(pair_count, 1))
    medians: list[torch.Tensor] = []
    for start in range(0, sample.shape[1], columns_per_batch):
        chunk = sample[:, start : start + columns_per_batch]
        distances = chunk.index_select(0, pairs[0]).sub(chunk.index_select(0, pairs[1])).abs_()
        medians.append(distances.median(dim=0).values)
    robust = torch.cat(medians).clamp_min(lower_bound)
    numeric.lengthscale = robust.reshape(1, -1)
