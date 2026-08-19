# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch

from leanhebo import LeanHEBO
from leanhebo.acquisition import PosteriorEvaluator, PosteriorStats
from leanhebo.config import (
    AcquisitionConfig,
    GPConfig,
    LeanHEBOConfig,
    RuntimeConfig,
    SearchConfig,
    WarpConfig,
)
from leanhebo.data import CandidateBatch, EncodedBatch
from leanhebo.errors import NumericalError, SearchSpaceExhaustedError
from leanhebo.gp import ExactGPSurrogate
from leanhebo.space import Bool, Categorical, CompiledSpace, FixedInput, Float, Integer, Space


def _space() -> Space:
    return Space(
        Float("x", -2.0, 2.0),
        Integer("depth", 2, 10, step=2),
        Categorical("kind", ("a", "b", "c")),
        Bool("enabled"),
    )


def _config(
    *,
    seed: int = 17,
    dtype: Literal["float32", "float64"] = "float32",
    acquisition_batch_size: int = 3,
    generations: int = 1,
) -> LeanHEBOConfig:
    return LeanHEBOConfig(
        random_samples=3,
        runtime=RuntimeConfig(
            device="cpu",
            dtype=dtype,
            seed=seed,
            acquisition_batch_size=acquisition_batch_size,
        ),
        gp=GPConfig(
            optimizer="adam",
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
        ),
        warp=WarpConfig(method="none"),
        acquisition=AcquisitionConfig(stochastic=True),
        search=SearchConfig(
            population_size=8,
            generations=generations,
            eliminate_duplicates=False,
            seed=seed + 1,
        ),
    )


def _outcomes(candidates: CandidateBatch) -> torch.Tensor:
    continuous = candidates.continuous
    categorical = candidates.categorical
    return continuous.square().sum(dim=1) + categorical.to(continuous.dtype).sum(dim=1)


def test_optimizer_observes_its_candidate_batch_without_reencoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(dtype="float64"))
    candidates = optimizer.suggest(3)
    outcomes = _outcomes(candidates)
    expected_continuous = candidates.continuous.clone()
    expected_categorical = candidates.categorical.clone()
    expected_outcomes = outcomes.clone()

    def fail_encode(self: CompiledSpace, value: object) -> EncodedBatch:
        raise AssertionError(f"unexpected re-encoding of {type(value).__name__}")

    monkeypatch.setattr(CompiledSpace, "encode", fail_encode)
    assert optimizer.observe(candidates, outcomes) == len(candidates)
    stored = optimizer.store.materialize()
    assert stored.continuous.data_ptr() != candidates.continuous.data_ptr()
    assert stored.categorical.data_ptr() != candidates.categorical.data_ptr()

    candidates.continuous.add_(10)
    candidates.categorical.zero_()
    outcomes.add_(10)

    actual = optimizer.store.materialize()
    assert torch.equal(actual.continuous, expected_continuous)
    assert torch.equal(actual.categorical, expected_categorical)
    assert torch.equal(actual.y, expected_outcomes)
    assert not optimizer.store.unique_mask(
        EncodedBatch(expected_continuous, expected_categorical)
    ).any()


def test_finite_space_exhaustion_never_returns_duplicate_candidates() -> None:
    optimizer = LeanHEBO(
        Space(Bool("flag")),
        config=LeanHEBOConfig(runtime=RuntimeConfig(seed=1)),
    )

    with pytest.raises(SearchSpaceExhaustedError, match=r"3 unique.*only 2 unseen"):
        optimizer.suggest(3)

    assert optimizer.diagnostics.counters["suggest.uniqueness_exhausted"] == 1


def test_sequential_initial_suggestions_follow_the_same_sobol_prefix_as_a_batch() -> None:
    config = LeanHEBOConfig(
        random_samples=64,
        runtime=RuntimeConfig(seed=23),
    )
    space = Space(Float("x", -1.0, 1.0))
    batched_optimizer = LeanHEBO(space, config=config)
    sequential_optimizer = LeanHEBO(space, config=config)

    batched = batched_optimizer.suggest(6)
    sequential_rows: list[CandidateBatch] = []
    for _ in range(6):
        candidate = sequential_optimizer.suggest(1)
        sequential_rows.append(candidate)
        sequential_optimizer.observe(candidate, torch.zeros(1))

    sequential = torch.cat([candidate.continuous for candidate in sequential_rows])
    assert torch.equal(sequential, batched.continuous)
    assert sequential_optimizer._sobol_draw_count == batched_optimizer._sobol_draw_count == 6


def test_discrete_initial_suggestions_are_independent_of_batching() -> None:
    config = LeanHEBOConfig(
        random_samples=64,
        runtime=RuntimeConfig(seed=2),
    )
    space = Space(Integer("index", 0, 3), Categorical("kind", ("a", "b", "c")))
    batched_optimizer = LeanHEBO(space, config=config)
    sequential_optimizer = LeanHEBO(space, config=config)

    batched = batched_optimizer.suggest(8)
    sequential_records: list[dict[str, object]] = []
    for _ in range(8):
        candidate = sequential_optimizer.suggest(1)
        sequential_records.extend(candidate.to_records())
        sequential_optimizer.observe(candidate, torch.zeros(1))

    assert sequential_records == batched.to_records()
    assert sequential_optimizer._sobol_draw_count == batched_optimizer._sobol_draw_count == 8


def test_finite_context_completion_finds_the_last_unseen_integer_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = LeanHEBO(
        Space(
            Integer("stepped", 2, 4, step=2),
            Integer("logarithmic", 1, 2, log=True, base=2),
            Integer("exponent", 1, 2, exponent=True, base=2),
            Float("fixed_float", -1.0, 1.0),
            Categorical("context", ("a", "b")),
            Bool("flag"),
        ),
        config=LeanHEBOConfig(
            random_samples=9,
            runtime=RuntimeConfig(seed=5),
        ),
    )
    records = [
        {
            "stepped": stepped,
            "logarithmic": logarithmic,
            "exponent": exponent,
            "fixed_float": 0.25,
            "context": "b",
            "flag": True,
        }
        for stepped in (2, 4)
        for logarithmic in (1, 2)
        for exponent in (1, 2)
    ]
    observed = optimizer.space.decode(optimizer.space.encode(records[:-1]))
    optimizer.observe(observed, torch.arange(7, dtype=optimizer.dtype))

    def repeated_observed_draw(count: int, fixed: FixedInput | None) -> CandidateBatch:
        encoded = optimizer.space.encode([records[0]] * count).to(
            optimizer.device,
            dtype=optimizer.dtype,
        )
        return optimizer.space.decode(encoded, fixed=fixed)

    monkeypatch.setattr(optimizer, "_draw_sobol", repeated_observed_draw)

    assert optimizer.suggest(
        1,
        fix_input={"fixed_float": 0.25, "context": "b", "flag": True},
    ).to_records() == [records[-1]]


def test_integrated_mace_evaluates_each_candidate_chunk_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _config(acquisition_batch_size=3, generations=1)
    optimizer = LeanHEBO(
        _space(),
        config=replace(base, search=replace(base.search, population_size=4)),
    )
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    events: list[tuple[int, int]] = []
    original_evaluate = PosteriorEvaluator.evaluate

    def recording_evaluate(
        evaluator: PosteriorEvaluator,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> PosteriorStats:
        provider = evaluator.provider
        before = provider.posterior_calls  # type: ignore[attr-defined]
        result = original_evaluate(evaluator, continuous, categorical)
        after = provider.posterior_calls  # type: ignore[attr-defined]
        events.append((len(continuous), after - before))
        return result

    monkeypatch.setattr(PosteriorEvaluator, "evaluate", recording_evaluate)
    optimizer.suggest(2, fix_input={"kind": "b", "enabled": True})

    assert optimizer.surrogate is not None
    assert len(events) >= optimizer.config.search.generations + 2
    assert events[0] == (1, 1)  # contextual incumbent
    search_events = events[1 : optimizer.config.search.generations + 2]
    assert [count for count, _ in search_events] == [4, 4]
    for candidate_count, posterior_calls in events:
        assert posterior_calls == math.ceil(candidate_count / 3)
    assert optimizer.surrogate.posterior_calls == sum(calls for _, calls in events)
    assert optimizer.diagnostics.counters["posterior.calls"] == sum(calls for _, calls in events)


def test_posterior_numerical_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(generations=0))
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    calls = 0

    def fail(
        surrogate: ExactGPSurrogate,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del surrogate, continuous, categorical
        nonlocal calls
        calls += 1
        raise NumericalError("synthetic posterior failure")

    monkeypatch.setattr(ExactGPSurrogate, "predict", fail)
    with pytest.raises(NumericalError, match="synthetic posterior failure"):
        optimizer.suggest(1)

    assert calls == 1
    assert optimizer.surrogate is not None
    assert optimizer.surrogate.full_refit_count == 1


def test_optimizer_uses_fantasy_update_when_preprocessing_and_cache_are_stable() -> None:
    optimizer = LeanHEBO(
        Space(Categorical("choice", tuple("abcdefgh"))),
        config=LeanHEBOConfig(
            random_samples=2,
            runtime=RuntimeConfig(seed=41, acquisition_batch_size=None),
            gp=GPConfig(
                optimizer="adam",
                initial_steps=1,
                update_steps=0,
                full_refit_interval=None,
                full_refit_growth_factor=None,
                use_fantasy_updates=True,
            ),
            warp=WarpConfig(method="none", refit_interval=3),
            search=SearchConfig(
                population_size=4,
                generations=0,
                eliminate_duplicates=False,
                seed=42,
            ),
        ),
    )
    initial = optimizer.suggest(2)
    optimizer.observe(initial, _outcomes(initial))
    first_model_candidate = optimizer.suggest(1)
    optimizer.observe(first_model_candidate, _outcomes(first_model_candidate))
    assert optimizer.surrogate is not None and optimizer.surrogate.model is not None
    assert optimizer.surrogate.model.prediction_strategy is not None

    second_model_candidate = optimizer.suggest(1)

    assert len(second_model_candidate) == 1
    assert optimizer.diagnostics.counters["gp.fantasy_update"] == 1
    assert optimizer.diagnostics.fit_reports[-1].kind == "fantasy_update"
    assert optimizer.surrogate.train_targets is not None
    assert optimizer.surrogate.train_targets.shape == (3,)


@pytest.mark.parametrize(
    ("configured_dtype", "torch_dtype"),
    [("float32", torch.float32), ("float64", torch.float64)],
)
def test_cpu_dtype_propagates_through_store_gp_search_and_candidates(
    configured_dtype: Literal["float32", "float64"],
    torch_dtype: torch.dtype,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(dtype=configured_dtype, generations=0))
    initial = optimizer.suggest(3)
    assert initial.device.type == "cpu"
    assert initial.dtype == torch_dtype
    assert initial.categorical.dtype == torch.int64
    optimizer.observe(initial, _outcomes(initial))

    model_candidates = optimizer.suggest(2)
    assert model_candidates.dtype == torch_dtype
    assert optimizer.store.continuous.dtype == torch_dtype
    assert optimizer.store.y.dtype == torch_dtype
    assert optimizer.surrogate is not None
    assert optimizer.surrogate.dtype == torch_dtype
    assert optimizer.surrogate.train_continuous is not None
    assert optimizer.surrogate.train_continuous.dtype == torch_dtype
    assert optimizer.surrogate.train_targets is not None
    assert optimizer.surrogate.train_targets.dtype == torch_dtype
    assert optimizer.surrogate.model is not None
    assert all(
        parameter.dtype == torch_dtype for parameter in optimizer.surrogate.model.parameters()
    )
    assert optimizer.last_search is not None
    assert optimizer.last_search.population.dtype == torch_dtype
    assert optimizer.last_search.objectives.dtype == torch_dtype
    assert optimizer._previous_population is None


def test_reused_search_population_does_not_alias_last_search() -> None:
    base = _config(generations=0)
    config = replace(
        base,
        search=replace(base.search, reuse_previous_population=True),
    )
    optimizer = LeanHEBO(_space(), config=config)
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    optimizer.suggest(1)
    assert optimizer.last_search is not None
    assert optimizer._previous_population is not None
    expected = optimizer._previous_population.clone()

    optimizer.last_search.population.add_(123)

    torch.testing.assert_close(optimizer._previous_population, expected)


def test_checkpoint_after_observe_preserves_stale_model_continuation(
    tmp_path: Path,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(seed=29, generations=1))
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    first_model_batch = optimizer.suggest(2)
    optimizer.observe(first_model_batch, _outcomes(first_model_batch))
    assert optimizer.surrogate is not None
    assert optimizer._model_observation_version != optimizer.store.observation_version

    checkpoint = tmp_path / "stale-model.leanhebo"
    optimizer.save(checkpoint)
    restored = LeanHEBO.load(checkpoint, map_location="cpu")
    expected = optimizer.suggest(3, fix_input={"kind": "c"})
    actual = restored.suggest(3, fix_input={"kind": "c"})

    assert torch.equal(actual.continuous, expected.continuous)
    assert torch.equal(actual.categorical, expected.categorical)
    assert actual.to_records() == expected.to_records()
    assert restored.surrogate is not None
    assert restored.surrogate.train_targets is not None
    assert restored.surrogate.train_targets.shape == (5,)
