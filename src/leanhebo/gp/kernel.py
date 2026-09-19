# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""HEBO-compatible mixed numeric/categorical feature and kernel construction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import gpytorch  # type: ignore[import-untyped]
import torch
from torch import nn

_PAIRWISE_DISTANCE_ELEMENT_BUDGET = 8_000_000


class _MaternKernel(gpytorch.kernels.MaternKernel):  # type: ignore[misc]
    """Matérn-3/2 with stable distances at small learned lengthscales."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(nu=1.5, **kwargs)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: object,
    ) -> torch.Tensor:
        del params
        # Float32 quadratic distance expansion can corrupt even self-distances.
        # Normalize in double before direct distances to preserve nearby points.
        lengthscale = self.lengthscale.double()
        left = x1.double() / lengthscale
        right = x2.double() / lengthscale
        if last_dim_is_batch:
            left = left.transpose(-1, -2).unsqueeze(-1)
            right = right.transpose(-1, -2).unsqueeze(-1)
        distance: torch.Tensor
        if diag:
            distance = torch.linalg.vector_norm(left - right, dim=-1)
        elif left.is_cuda and not torch.is_grad_enabled():
            # Direct CUDA cdist is slow for the small feature blocks used here.
            # Bound the broadcast temporary; training retains cdist's lean backward.
            batch_shape = torch.broadcast_shapes(  # type: ignore[no-untyped-call]
                left.shape[:-2], right.shape[:-2]
            )
            batch_size = math.prod(batch_shape)
            elements_per_row = batch_size * right.shape[-2] * left.shape[-1]
            chunk_rows = max(1, _PAIRWISE_DISTANCE_ELEMENT_BUDGET // max(1, elements_per_row))
            chunks = [
                torch.linalg.vector_norm(chunk.unsqueeze(-2) - right.unsqueeze(-3), dim=-1)
                for chunk in left.split(chunk_rows, dim=-2)
            ]
            distance = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-2)
        else:
            distance = torch.cdist(left, right, compute_mode="donot_use_mm_for_euclid_dist")
        scaled = math.sqrt(3) * distance
        return ((1 + scaled) * torch.exp(-scaled)).to(dtype=x1.dtype)


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
            _MaternKernel(
                ard_num_dims=num_continuous if ard else None,
                active_dims=torch.arange(num_continuous),
            )
        )
    if feature_extractor.embedding_sizes:
        components.append(
            _MaternKernel(
                active_dims=torch.arange(num_continuous, feature_extractor.output_dimensions),
            )
        )
    if not components:
        raise ValueError("an exact GP requires at least one non-fixed input dimension")
    return components[0] if len(components) == 1 else gpytorch.kernels.ProductKernel(*components)


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
