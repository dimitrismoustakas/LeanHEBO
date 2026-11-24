# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch

from leanhebo import LeanHEBO
from leanhebo.acquisition import MACEEvaluator, PosteriorEvaluator, PosteriorStats
from leanhebo.checkpoint import save_checkpoint
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
from leanhebo.space import Bool, Categorical, CompiledSpace, Eq, FixedInput, Float, Integer, Space


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


def test_fully_static_space_returns_its_only_point_then_exhausts() -> None:
    optimizer = LeanHEBO(
        Space(Integer("fold", 3, 3), Categorical("dataset", ("protein",))),
        config=LeanHEBOConfig(runtime=RuntimeConfig(seed=1)),
    )

    assert optimizer.random_samples == 1
    candidate = optimizer.suggest(1)
    assert candidate.to_records() == [{"fold": 3, "dataset": "protein"}]
    assert candidate.continuous.shape == (1, 0)
    assert candidate.categorical.shape == (1, 0)
    optimizer.observe(candidate, torch.tensor([0.0]))

    with pytest.raises(SearchSpaceExhaustedError, match=r"only 0 unseen"):
        optimizer.suggest(1)

    assert optimizer.surrogate is None
    assert optimizer.refit() is None


def test_static_category_is_not_fitted_as_a_gp_embedding() -> None:
    optimizer = LeanHEBO(
        Space(Float("x", -1.0, 1.0), Categorical("dataset", ("protein",))),
        config=_config(generations=0),
    )
    initial = optimizer.suggest(3)
    optimizer.observe(initial, initial.continuous[:, 0].square())

    candidate = optimizer.suggest(1)

    assert candidate.to_records()[0]["dataset"] == "protein"
    assert optimizer.surrogate is not None
    assert optimizer.surrogate.category_sizes == ()
    assert optimizer.surrogate.train_categorical is not None
    assert optimizer.surrogate.train_categorical.shape == (3, 0)


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
        counters = evaluator.provider.diagnostics.counters  # type: ignore[attr-defined]
        before = counters["posterior.calls"]
        result = original_evaluate(evaluator, continuous, categorical)
        events.append((len(continuous), counters["posterior.calls"] - before))
        return result

    monkeypatch.setattr(PosteriorEvaluator, "evaluate", recording_evaluate)
    optimizer.suggest(2, fix_input={"kind": "b", "enabled": True})

    assert optimizer.surrogate is not None
    assert len(events) == optimizer.config.search.generations + 2
    assert events[0] == (1, 1)  # contextual incumbent
    search_events = events[1 : optimizer.config.search.generations + 2]
    assert [count for count, _ in search_events] == [4, 4]
    for candidate_count, posterior_calls in events:
        assert posterior_calls == math.ceil(candidate_count / 3)
    assert optimizer.diagnostics.counters["posterior.calls"] == sum(calls for _, calls in events)


def test_default_search_scores_every_legal_log_integer_once() -> None:
    optimizer = LeanHEBO(
        Space(Integer("x", 3, 19, log=True), Bool("flag")),
        config=replace(_config(), search=SearchConfig()),
    )
    optimizer.observe([{"x": v, "flag": False} for v in (3, 8, 18)], [0.0, 1.0, 2.0])
    optimizer.suggest(2)
    result = optimizer.last_search
    assert result is not None
    assert (result.generations, result.objective_calls, result.candidate_evaluations) == (0, 1, 34)
    rows = optimizer.space.candidate_from_dense(result.population).to_records()
    assert {(row["x"], row["flag"]) for row in rows} == {
        (x, flag) for x in range(3, 20) for flag in (False, True)
    }


def test_automatic_conditional_search_replays_one_full_native_population() -> None:
    space = Space(Bool("on"), Integer("x", 1, 5, active_when=Eq("on", True)), Integer("k", 0, 2))
    config = replace(_config(), search=SearchConfig())
    automatic = LeanHEBO(space, config=config)
    explicit = LeanHEBO(
        space, config=replace(config, search=SearchConfig(population_size=18, generations=0))
    )
    for count in (1, 1, 1, 1, 3, 1):
        a, b = automatic.suggest(count), explicit.suggest(count)
        assert a.to_records() == b.to_records()
        loss = a.continuous.sum(dim=1)
        automatic.observe(a, loss)
        explicit.observe(b, loss)
    assert automatic.last_search is not None and automatic.last_search.candidate_evaluations == 18


def test_exact_search_uses_global_front_then_sobol(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = LeanHEBO(
        Space(Integer("x", 0, 20)), config=replace(_config(), search=SearchConfig())
    )
    optimizer.observe([{"x": v} for v in (0, 1, 2)], [0.0, 1.0, 2.0])

    # The only global front member has already been measured.
    def scores(
        self: MACEEvaluator, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        return continuous.expand(-1, 3)

    monkeypatch.setattr(MACEEvaluator, "__call__", scores)
    assert len(optimizer.suggest(2)) == 2
    assert optimizer._sobol_draw_count == 2


def test_exhaustive_cutoff_uses_the_current_conditional_context() -> None:
    space = Space(
        Bool("enabled"),
        Integer("choice", 1, 4, active_when=Eq("enabled", False)),
        Float("rate", 0, 1, active_when=Eq("enabled", True)),
    )
    optimizer = LeanHEBO(
        space,
        config=replace(
            _config(),
            search=SearchConfig(
                population_size=3,
                generations=1,
                exhaustive_limit=4,
            ),
        ),
    )
    optimizer.observe(
        [
            {"enabled": True, "rate": 0.2},
            {"enabled": True, "rate": 0.5},
            {"enabled": False, "choice": 2},
        ],
        [0.3, 0.5, 0.1],
    )
    optimizer.suggest(fix_input={"enabled": False})
    result = optimizer.last_search
    assert result is not None and result.candidate_evaluations == 4 and result.generations == 0
    # Fixing a child leaves its parent free, giving four discrete points plus one active rate.
    optimizer.suggest(fix_input={"rate": 0.7})
    assert optimizer.last_search is not None and optimizer.last_search.generations == 1
    optimizer.suggest(fix_input={"enabled": False, "choice": 1})
    result = optimizer.last_search
    assert result is not None and result.candidate_evaluations == 1


def test_encoded_only_fixed_context_does_not_reuse_the_unrestricted_grid() -> None:
    opt = LeanHEBO(
        Space(Bool("flag"), Integer("x", 1, 4)), config=replace(_config(), search=SearchConfig())
    )
    opt.observe([{"flag": False, "x": x} for x in (1, 2, 3)], [1.0, 2.0, 3.0])
    opt.suggest(fix_input=replace(opt.space.compile_fixed({"flag": True}), decoded_values=()))
    assert opt.last_search is not None and opt.last_search.candidate_evaluations == 4
    opt.suggest()
    assert opt.last_search is not None and opt.last_search.candidate_evaluations == 8


def test_automatic_finite_cutoff_is_inclusive_and_survives_checkpoint(tmp_path: Path) -> None:
    config = replace(_config(acquisition_batch_size=4096), search=SearchConfig())
    optimizer = LeanHEBO(Space(Integer("x", 1, 8192)), config=config)
    optimizer.observe([{"x": v} for v in (2, 23, 184)], [0.1, 1.0, 0.5])
    path = tmp_path / "finite.leanhebo"
    optimizer.save(path)
    restored = LeanHEBO.load(path)
    assert restored.suggest().to_records() == optimizer.suggest().to_records()
    result = restored.last_search
    assert result is not None and (result.generations, result.candidate_evaluations) == (0, 8192)
    above = LeanHEBO(Space(Integer("x", 0, 8192)), config=config)
    assert above._finite_pool(None) is None


@pytest.mark.parametrize("previous_limit", [0, 2048])
def test_old_checkpoints_resume_evolutionary_search(tmp_path: Path, previous_limit: int) -> None:
    space = Space(Integer("x", 0, max(199, previous_limit)))
    config = replace(
        _config(acquisition_batch_size=4096), search=SearchConfig(exhaustive_limit=previous_limit)
    )
    optimizer = LeanHEBO(space, config=config)
    optimizer.observe([{"x": v} for v in (2, 23, 184)], [0.1, 1.0, 0.5])
    state = optimizer._checkpoint_state()
    if previous_limit:
        state["config"]["search"]["exhaustive_limit"] = None
    else:
        del state["config"]["search"]["exhaustive_limit"]
    path = tmp_path / "old.leanhebo"
    save_checkpoint(path, state)
    restored = LeanHEBO.load(path)
    assert restored.config.search.exhaustive_limit == previous_limit
    assert restored.suggest().to_records() == optimizer.suggest().to_records()
    assert restored.last_search is not None and restored.last_search.generations > 0


def test_batch_uses_pareto_fronts_of_the_unseen_population(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = LeanHEBO(
        Space(Integer("x", 0, 6)),
        config=LeanHEBOConfig(
            random_samples=2,
            runtime=RuntimeConfig(seed=7),
            gp=GPConfig(initial_steps=0),
            search=SearchConfig(population_size=7, generations=1),
        ),
    )
    optimizer.observe([{"x": 0}, {"x": 6}], [0.0, 6.0])
    # 0 dominates 6, which dominates 2. Removing both observed points promotes
    # 2 from the third front to the first, alongside 1 and ahead of 3.
    scores = torch.tensor(
        [[0, 1, 0], [2, 0, 0], [1, 2, 0], [3, 0, 0], [3, 3, 3], [4, 4, 4], [0, 2, 0]],
        dtype=optimizer.dtype,
    )

    def objective(
        self: MACEEvaluator, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        return scores[continuous[:, 0].long()]

    monkeypatch.setattr(MACEEvaluator, "__call__", objective)
    assert optimizer.suggest(1).to_records() == [{"x": 1}]
    for count in (2, 3):
        assert {row["x"] for row in optimizer.suggest(count).to_records()} == set(
            range(1, count + 1)
        )
    assert optimizer._sobol_draw_count == 0


@pytest.mark.parametrize("parameter", [Float("x", 0.0, 1.0), Integer("x", 0, 1000)])
def test_batch_explores_beyond_a_nonexhausted_population(
    monkeypatch: pytest.MonkeyPatch, parameter: Float | Integer
) -> None:
    optimizer = LeanHEBO(
        Space(parameter),
        config=LeanHEBOConfig(
            random_samples=2,
            runtime=RuntimeConfig(seed=7),
            gp=GPConfig(initial_steps=0),
            search=SearchConfig(population_size=7, generations=1),
        ),
    )
    optimizer.observe([{"x": 0}, {"x": 1}], [0.0, 1.0])

    def objective(
        self: MACEEvaluator, continuous: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        # The observed incumbent is the only non-dominated point.
        return continuous[:, :1].expand(-1, 3)

    monkeypatch.setattr(MACEEvaluator, "__call__", objective)
    assert len(optimizer.suggest(3)) == 3
    assert optimizer.last_search is not None and optimizer.last_search.generations == 1
    assert optimizer._sobol_draw_count == 3


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
    assert optimizer.diagnostics.counters["gp.initial"] == 1


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
    observations = optimizer.store.materialize()
    assert observations.continuous.dtype == torch_dtype
    assert observations.y.dtype == torch_dtype
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
