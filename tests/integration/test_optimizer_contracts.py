# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
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
from leanhebo.gp import ExactGPSurrogate, FitReport
from leanhebo.space import Bool, Categorical, CompiledSpace, Float, Integer, Space


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
    stored = optimizer.store.encoded_chunks[0]
    assert stored.continuous.data_ptr() != candidates.continuous.data_ptr()
    assert stored.categorical.data_ptr() != candidates.categorical.data_ptr()

    candidates.continuous.add_(10)
    candidates.categorical.zero_()
    outcomes.add_(10)

    assert torch.equal(stored.continuous, expected_continuous)
    assert torch.equal(stored.categorical, expected_categorical)
    assert torch.equal(optimizer.store.y_chunks[0], expected_outcomes)
    assert optimizer.store.contains(EncodedBatch(expected_continuous, expected_categorical)).all()


def test_finite_space_exhaustion_never_returns_duplicate_candidates() -> None:
    optimizer = LeanHEBO(
        Space(Bool("flag")),
        config=LeanHEBOConfig(runtime=RuntimeConfig(seed=1)),
    )

    with pytest.raises(SearchSpaceExhaustedError, match=r"3 unique.*only 2 unseen"):
        optimizer.suggest(3)

    assert optimizer.diagnostics.counters["suggest.uniqueness_exhausted"] == 1


def test_integrated_mace_evaluates_each_candidate_chunk_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = LeanHEBO(
        _space(),
        config=_config(acquisition_batch_size=3, generations=2),
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
    assert [count for count, _ in search_events] == [8, 8, 8]
    for candidate_count, posterior_calls in events:
        assert posterior_calls == math.ceil(candidate_count / 3)
    assert optimizer.surrogate.posterior_calls == sum(calls for _, calls in events)
    assert optimizer.diagnostics.counters["posterior.calls"] == sum(calls for _, calls in events)


def test_posterior_numerical_failure_gets_one_hard_refit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(generations=0))
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    original_predict = ExactGPSurrogate.predict
    calls = 0

    def fail_once(
        surrogate: ExactGPSurrogate,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise NumericalError("synthetic posterior failure")
        return original_predict(surrogate, continuous, categorical)

    monkeypatch.setattr(ExactGPSurrogate, "predict", fail_once)
    candidates = optimizer.suggest(1)

    assert len(candidates) == 1
    assert optimizer.surrogate is not None
    assert optimizer.surrogate.full_refit_count == 2
    assert optimizer.diagnostics.counters["gp.numerical_recovery_attempts"] == 1
    assert optimizer.diagnostics.counters["gp.numerical_recovery_failures"] == 0


def test_repeated_posterior_numerical_failure_raises_after_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(generations=0))
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    calls = 0

    def always_fail(
        surrogate: ExactGPSurrogate,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del surrogate, continuous, categorical
        nonlocal calls
        calls += 1
        raise NumericalError("persistent posterior failure")

    monkeypatch.setattr(ExactGPSurrogate, "predict", always_fail)
    with pytest.raises(NumericalError, match="persistent posterior failure"):
        optimizer.suggest(1)

    assert calls == 2
    assert optimizer.diagnostics.counters["gp.numerical_recovery_attempts"] == 1
    assert optimizer.diagnostics.counters["gp.numerical_recovery_failures"] == 1


def test_warm_fit_numerical_failure_gets_one_hard_refit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(generations=0))
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    appended = optimizer.suggest(1)
    optimizer.observe(appended, _outcomes(appended))
    original_fit = ExactGPSurrogate.fit
    calls = 0

    def fail_once(
        surrogate: ExactGPSurrogate,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        targets: torch.Tensor,
        *,
        transform_version: int,
        force_full_refit: bool = False,
        reset_parameters: bool = False,
    ) -> FitReport:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise NumericalError("synthetic warm-fit failure")
        assert reset_parameters
        return original_fit(
            surrogate,
            continuous,
            categorical,
            targets,
            transform_version=transform_version,
            force_full_refit=force_full_refit,
            reset_parameters=reset_parameters,
        )

    monkeypatch.setattr(ExactGPSurrogate, "fit", fail_once)
    candidates = optimizer.suggest(1)

    assert len(candidates) == 1
    assert calls == 2
    assert optimizer.surrogate is not None
    assert optimizer.surrogate.full_refit_count == 2
    assert optimizer.diagnostics.counters["gp.numerical_recovery_attempts"] == 1
    assert optimizer.diagnostics.counters["gp.numerical_recovery_failures"] == 0


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


def test_checkpoint_after_observe_preserves_stale_model_continuation(
    tmp_path: Path,
) -> None:
    optimizer = LeanHEBO(_space(), config=_config(seed=29, generations=1))
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _outcomes(initial))
    first_model_batch = optimizer.suggest(2)
    optimizer.observe(first_model_batch, _outcomes(first_model_batch))
    assert optimizer.surrogate is not None
    state = optimizer.state_dict()
    assert state["model_observation_version"] != state["observation_version"]

    checkpoint = tmp_path / "stale-model.leanhebo"
    optimizer.save(checkpoint)
    restored = LeanHEBO.load(checkpoint, map_location="cpu")
    expected = optimizer.suggest(3, fix_input={"kind": "c"})
    actual = restored.suggest(3, fix_input={"kind": "c"})

    assert torch.equal(actual.continuous, expected.continuous)
    assert torch.equal(actual.categorical, expected.categorical)
    assert actual.to_records() == expected.to_records()
    assert restored.surrogate is not None
    assert restored.surrogate.update_count == optimizer.surrogate.update_count
    assert restored.surrogate.train_targets is not None
    assert restored.surrogate.train_targets.shape == (5,)
