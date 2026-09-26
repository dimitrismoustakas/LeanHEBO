# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import pytest
import torch
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr

from leanhebo.acquisition import MACEEvaluator, PosteriorEvaluator, PosteriorStats


class _CountingPosterior:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del categorical
        self.calls += 1
        mean = continuous.sum(dim=-1)
        return mean, torch.full_like(mean, 0.25), mean.new_tensor(0.01)


def test_posterior_is_evaluated_once_per_chunk() -> None:
    provider = _CountingPosterior()
    evaluator = PosteriorEvaluator(provider, batch_size=4)
    continuous = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    categorical = torch.empty((7, 0), dtype=torch.long)
    result = evaluator.evaluate(continuous, categorical)
    assert provider.calls == 2
    assert result.mean.shape == (7,)


def test_empty_posterior_with_unbounded_chunk_size_is_well_defined() -> None:
    provider = _CountingPosterior()
    evaluator = PosteriorEvaluator(provider, batch_size=None)

    result = evaluator.evaluate(torch.empty((0, 2)), torch.empty((0, 0), dtype=torch.long))

    assert provider.calls == 0
    assert result.mean.shape == result.variance.shape == result.stddev.shape == (0,)
    assert result.noise_variance.shape == ()


def test_mace_is_finite_and_reproducible_with_dedicated_generator() -> None:
    continuous = torch.tensor([[0.1], [0.5], [0.9]])
    categorical = torch.empty((3, 0), dtype=torch.long)

    def evaluate(seed: int) -> torch.Tensor:
        provider = _CountingPosterior()
        posterior = PosteriorEvaluator(provider, batch_size=None)
        mace = MACEEvaluator(
            posterior,
            best_y=0.2,
            kappa=2.0,
            generator=torch.Generator().manual_seed(seed),
        )
        return mace(continuous, categorical)

    first = evaluate(11)
    second = evaluate(11)
    assert first.shape == (3, 3)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_mace_matches_pinned_upstream_fixed_seed_golden_trace() -> None:
    """Golden values were generated with HEBO ee6112d and Torch 2.13.0."""

    mean = torch.tensor([-5.0, 0.1, 2.0])
    variance = torch.tensor([0.01, 0.25, 1.0])
    stats = PosteriorStats(
        mean=mean,
        variance=variance,
        stddev=variance.sqrt(),
        noise_variance=torch.tensor(0.01),
    )
    evaluator = MACEEvaluator(
        PosteriorEvaluator(_CountingPosterior()),
        best_y=0.2,
        kappa=2.0,
        generator=torch.Generator().manual_seed(11),
    )

    actual = evaluator.from_stats(stats)
    expected = torch.tensor(
        [
            [-5.0956830978393555, -1.6834450960159302, -0.0],
            [-0.6248040199279785, 1.2124518156051636, 0.45422089099884033],
            [-0.09893012046813965, 4.1540656089782715, 3.24324107170105],
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_mace_deterministic_mode_does_not_advance_generator() -> None:
    provider = _CountingPosterior()
    posterior = PosteriorEvaluator(provider, batch_size=None)
    generator = torch.Generator().manual_seed(3)
    before = generator.get_state().clone()
    mace = MACEEvaluator(
        posterior,
        best_y=0.0,
        kappa=1.0,
        stochastic=False,
        generator=generator,
    )
    result = mace(torch.tensor([[0.0], [1.0]]), torch.empty((2, 0), dtype=torch.long))
    assert torch.equal(before, generator.get_state())
    assert torch.isfinite(result).all()


def _reference_log_ei(normalized: float, stddev: float) -> float:
    if normalized >= -1.0:
        density = math.exp(-0.5 * normalized**2) / math.sqrt(2.0 * math.pi)
        return math.log(stddev) + math.log(density + normalized * ndtr(normalized))
    # Integrate the positive EI density directly after scaling its tail to unit width.
    # This avoids subtracting Phi from phi and is independent of the erfcx implementation.
    magnitude = -normalized
    integral, _ = quad(
        lambda value: value * math.exp(-value - 0.5 * (value / magnitude) ** 2),
        0.0,
        math.inf,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    return (
        math.log(stddev)
        - 0.5 * normalized**2
        - 0.5 * math.log(2.0 * math.pi)
        - 2.0 * math.log(magnitude)
        + math.log(integral)
    )


def _deterministic_mace(normalized: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
    evaluator = MACEEvaluator(
        PosteriorEvaluator(_CountingPosterior()),
        best_y=0.0,
        kappa=2.0,
        epsilon=0.0,
        stochastic=False,
    )
    return evaluator.from_stats(
        PosteriorStats(
            mean=-normalized * stddev,
            variance=stddev.square(),
            stddev=stddev,
            noise_variance=normalized.new_zeros(()),
        )
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
            ],
        ),
    ],
)
def test_mace_log_improvement_matches_tail_integral_and_scipy(
    dtype: torch.dtype, device: str
) -> None:
    normalized = torch.tensor(
        [
            -1e8,
            -1.001e6,
            -1e6,
            -0.999e6,
            -1001.0,
            -1000.0,
            -999.0,
            -100.0,
            -20.0,
            -10.0,
            -6.001,
            -6.0,
            -5.999,
            -5.0,
            -4.501,
            -4.5,
            -2.0,
            -1.00001,
            -1.0,
            -0.99999,
            0.0,
            1.0,
            5.0,
            20.0,
            1e8,
        ],
        dtype=dtype,
        device=device,
    ).repeat(3)
    stddev = normalized.new_tensor([0.125, 1.0, 8.0]).repeat_interleave(normalized.numel() // 3)
    actual = _deterministic_mace(normalized, stddev)
    values = normalized.cpu().tolist()
    scales = stddev.cpu().tolist()
    expected = torch.tensor(
        [
            [-_reference_log_ei(value, scale), -float(log_ndtr(value))]
            for value, scale in zip(values, scales, strict=True)
        ],
        dtype=torch.float64,
        device=device,
    )
    tolerance = 5e-7 if dtype == torch.float32 else 5e-13
    torch.testing.assert_close(actual[:, 1:].double(), expected, rtol=tolerance, atol=tolerance * 4)
    torch.testing.assert_close(actual[:, 0], -normalized * stddev - 2.0 * stddev)
    assert actual.dtype == dtype and actual.device == normalized.device
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_improvement_objectives_are_monotone_across_tail_transitions(
    dtype: torch.dtype,
) -> None:
    tail_bound = 1e6 if dtype == torch.float64 else 1e3
    normalized = torch.cat(
        [
            torch.linspace(-20.0, 5.0, 25001, dtype=dtype),
            torch.linspace(-1.001, -0.999, 201, dtype=dtype),
            torch.linspace(-tail_bound * 1.001, -tail_bound * 0.999, 201, dtype=dtype),
            torch.tensor([-1e8, 1e8, 1e20], dtype=dtype),
        ]
    ).unique(sorted=True)
    actual = _deterministic_mace(normalized, torch.ones_like(normalized))

    assert torch.isfinite(actual).all()
    # Increasing z improves the mean at fixed variance, so neither objective may worsen.
    assert bool((actual[1:, 1:] <= actual[:-1, 1:]).all())
