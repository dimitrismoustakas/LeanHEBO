# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

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
