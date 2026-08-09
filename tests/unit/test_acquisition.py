# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from leanhebo.acquisition import MACEEvaluator, PosteriorEvaluator


class _CountingPosterior:
    posterior_cache_version = 0

    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del categorical
        self.calls += 1
        mean = continuous.sum(dim=-1)
        return mean, torch.full_like(mean, 0.25), mean.new_tensor(0.01)


def test_posterior_is_evaluated_once_per_chunk_and_cached() -> None:
    provider = _CountingPosterior()
    evaluator = PosteriorEvaluator(provider, batch_size=4, cache=True)
    continuous = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    categorical = torch.empty((7, 0), dtype=torch.long)
    first = evaluator.evaluate(continuous, categorical)
    second = evaluator.evaluate(continuous, categorical)
    assert provider.calls == 2
    assert first is second
    assert first.mean.shape == (7,)


def test_posterior_cache_distinguishes_views_with_shared_storage() -> None:
    provider = _CountingPosterior()
    evaluator = PosteriorEvaluator(provider, batch_size=None, cache=True)
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    categorical = torch.empty((2, 0), dtype=torch.long)

    rows = evaluator.evaluate(base, categorical)
    columns = evaluator.evaluate(base.mT, categorical)

    assert provider.calls == 2
    torch.testing.assert_close(rows.mean, torch.tensor([3.0, 7.0]))
    torch.testing.assert_close(columns.mean, torch.tensor([4.0, 6.0]))


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
        posterior = PosteriorEvaluator(provider, batch_size=None, cache=False)
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
