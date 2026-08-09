# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import ClassVar, cast

import pytest

from benchmarks.harness.results import BenchmarkResult
from benchmarks.harness.work import WorkBudget
from benchmarks.latency import fixed_history
from benchmarks.latency.compare_fixed_history import build_fixed_history_report
from benchmarks.latency.fixed_history import (
    FixedHistoryCell,
    make_history,
    make_objective,
    matrix_cells,
    run_fixed_history,
)
from benchmarks.quality.compare_results import RawResult
from benchmarks.quality.objectives import ToyObjective
from benchmarks.quality.runner import RunSettings, Suggested


def _settings(cell: FixedHistoryCell) -> RunSettings:
    return RunSettings(
        evaluation_budget=cell.observations,
        batch_size=cell.batch_size,
        random_samples=4,
        population_size=4,
        generations=0,
        gp_initial_steps=0,
        gp_update_steps=0,
        posterior_batch_size=None,
        model_lifecycle="cold",
    )


def test_declared_matrix_expands_to_every_requested_cell() -> None:
    cells = matrix_cells(
        ("continuous", "mixed"),
        (5, 20),
        (16, 64, 128),
        (1, 4),
    )

    assert len(cells) == 24
    assert cells[0].name == "continuous-d5-n16-q1"
    assert cells[-1].name == "mixed-d20-n128-q4"
    with pytest.raises(ValueError, match="unsupported dimensions"):
        matrix_cells(("continuous",), (7,), (16,), (1,))
    with pytest.raises(ValueError, match="duplicate selections"):
        matrix_cells(("continuous",), (5, 5), (16,), (1,))
    with pytest.raises(ValueError, match="dimension must be one of"):
        FixedHistoryCell("continuous", dimension=7, observations=16, batch_size=1)


def test_fixed_history_is_deterministic_valid_and_case_specific() -> None:
    cell = FixedHistoryCell("mixed", dimension=5, observations=16, batch_size=1)
    objective = make_objective(cell)
    first = make_history(cell, seed=7)
    repeated = make_history(cell, seed=7)
    another_seed = make_history(cell, seed=8)

    assert [parameter.kind for parameter in objective.parameters] == [
        "float",
        "integer",
        "categorical",
        "float",
        "integer",
    ]
    assert first == repeated
    assert first.sha256 != another_seed.sha256
    assert len(first.rows) == len(first.values) == 16
    assert len(first.sha256) == 64
    assert all(value >= 0 for value in first.values)


class _FixedAdapter:
    implementation: ClassVar[dict[str, object]] = {
        "name": "fixed-double",
        "version": "test",
        "commit": None,
    }

    def __init__(self, settings: RunSettings) -> None:
        self.work = WorkBudget(
            objective_evaluations=settings.evaluation_budget,
            batch_size=settings.batch_size,
            population_size=settings.population_size,
            generations=settings.generations,
            gp_initial_steps=settings.gp_initial_steps,
            gp_update_steps=settings.gp_initial_steps,
            full_refit_interval=1,
            posterior_batch_size=None,
            random_samples=settings.random_samples,
            search_candidate_evaluations=settings.population_size * (settings.generations + 1),
            gp_optimizer=settings.gp_optimizer,
            learning_rate=settings.learning_rate,
            reuse_parameters=False,
            reuse_optimizer_state=False,
            use_set_train_data=False,
            device="cpu",
            dtype="float32",
            torch_threads=settings.torch_threads,
        )
        self._batch_size = settings.batch_size
        self._generations = settings.generations
        self._candidate_evaluations = settings.population_size * (settings.generations + 1)
        self.observed = 0

    def observe(self, suggested: Suggested, values: Sequence[float]) -> None:
        assert isinstance(suggested.native, list)
        assert len(suggested.rows) == len(values)
        self.observed += len(values)

    def suggest(self, count: int) -> Suggested:
        assert self.observed == self.work.objective_evaluations
        assert count == self._batch_size
        rows = [
            {f"x{dimension}": float(row + dimension) / 10 for dimension in range(5)}
            for row in range(count)
        ]
        return Suggested(
            rows,
            native=None,
            search_report={
                "objective_calls": self._generations + 1,
                "candidate_evaluations": self._candidate_evaluations,
                "offspring_generations": self._generations,
            },
        )

    def phase_wall_times(self) -> Mapping[str, Sequence[float]]:
        return {}

    def metrics(self) -> Mapping[str, object]:
        return {
            "observations": self.observed,
            "numerical_stability": {
                "jitter": {"escalations": 1, "maximum_parameter": 0.01},
                "fit": {"give_ups": 0},
                "prediction": {"random_fallbacks": 1},
            },
        }


def test_fixed_history_repeats_reconstruct_and_validate_the_same_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = FixedHistoryCell("continuous", dimension=5, observations=16, batch_size=4)
    settings = _settings(cell)
    adapters: list[_FixedAdapter] = []

    def fake_make_adapter(
        implementation: str,
        objective: ToyObjective,
        received_settings: RunSettings,
        seed: int,
    ) -> _FixedAdapter:
        assert implementation == "leanhebo"
        assert seed == 3
        assert objective.name == "fixed-continuous-d5"
        adapter = _FixedAdapter(received_settings)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(fixed_history, "make_adapter", fake_make_adapter)
    result = run_fixed_history("leanhebo", cell, settings, seed=3, repeats=2)

    assert len(adapters) == 2
    assert result.failures == []
    assert len(result.phases[fixed_history.SUGGEST_PHASE]["wall_seconds"]) == 2
    assert result.metrics["repeats_completed"] == 2
    assert result.metrics["history_observations"] == 16
    assert result.metrics["suggestions_returned"] == 8
    assert result.metrics["invalid_suggestions"] == 0
    assert result.metrics["duplicate_suggestions"] == 0
    grouped_metrics = cast(dict[str, object], result.metrics["implementation_metrics"])
    assert len(cast(list[object], grouped_metrics["repeats"])) == 2
    numerical = cast(dict[str, dict[str, object]], grouped_metrics["numerical_stability"])
    assert numerical["jitter"] == {"escalations": 2, "maximum_parameter": 0.01}
    assert numerical["prediction"] == {"random_fallbacks": 2}
    assert result.quality == {"normalized_regret": None}


class _VaryingAdapter(_FixedAdapter):
    def __init__(self, settings: RunSettings, offset: int) -> None:
        super().__init__(settings)
        self._offset = offset

    def suggest(self, count: int) -> Suggested:
        suggested = super().suggest(count)
        suggested.rows[0]["x0"] = float(suggested.rows[0]["x0"]) + self._offset
        return suggested


def test_fixed_history_flags_nondeterministic_repeat_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = FixedHistoryCell("continuous", dimension=5, observations=16, batch_size=1)
    settings = _settings(cell)
    created = 0

    def fake_make_adapter(
        implementation: str,
        objective: ToyObjective,
        received_settings: RunSettings,
        seed: int,
    ) -> _VaryingAdapter:
        nonlocal created
        adapter = _VaryingAdapter(received_settings, created)
        created += 1
        return adapter

    monkeypatch.setattr(fixed_history, "make_adapter", fake_make_adapter)
    result = run_fixed_history("leanhebo", cell, settings, seed=3, repeats=2)

    assert result.metrics["repeats_completed"] == 2
    assert result.failures[-1]["stage"] == "repeat-determinism"


def test_fixed_history_rejects_non_cold_or_misaligned_settings() -> None:
    cell = FixedHistoryCell("continuous", dimension=5, observations=16, batch_size=1)

    with pytest.raises(ValueError, match="history size"):
        run_fixed_history(
            "leanhebo",
            cell,
            RunSettings(
                evaluation_budget=17,
                batch_size=1,
                model_lifecycle="cold",
            ),
            seed=0,
        )
    with pytest.raises(ValueError, match="cold model lifecycle"):
        run_fixed_history(
            "leanhebo",
            cell,
            RunSettings(
                evaluation_budget=16,
                batch_size=1,
                model_lifecycle="persistent",
            ),
            seed=0,
        )


def test_fixed_history_comparator_rejects_different_preloaded_data() -> None:
    cell = FixedHistoryCell("continuous", dimension=5, observations=16, batch_size=1)
    work = _FixedAdapter(_settings(cell)).work
    common = BenchmarkResult(
        implementation={"name": "leanhebo", "version": "test", "commit": "a" * 40},
        suite="fixed-history-latency",
        case=cell.name,
        seed=5,
        work=work,
        phases={
            fixed_history.SUGGEST_PHASE: {
                "wall_seconds": [1.0],
                "process_cpu_seconds": [1.0],
            }
        },
        metrics={
            "history_sha256": "1" * 64,
            "history_observations": 16,
            "public_dimension": 5,
            "family": "continuous",
            "repeats_requested": 1,
            "duplicate_suggestions": 0,
        },
        quality={"normalized_regret": None},
    ).to_dict()
    candidate = cast(RawResult, common)
    baseline = deepcopy(candidate)
    baseline["implementation"] = {
        "name": "upstream-hebo",
        "version": "test",
        "commit": "b" * 40,
    }
    key = ("fixed-history-latency", cell.name, 5)

    report = build_fixed_history_report({key: candidate}, {key: baseline})
    assert report["comparison_lane"] == "matched-work"

    baseline["metrics"]["history_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="refusing fixed-history comparison"):
        build_fixed_history_report({key: candidate}, {key: baseline})
