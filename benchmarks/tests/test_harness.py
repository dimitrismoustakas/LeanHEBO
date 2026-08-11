# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

from benchmarks.environments.prepare_upstream import UpstreamManifest
from benchmarks.harness.results import BenchmarkResult, PhaseRecorder, write_result
from benchmarks.harness.work import WorkBudget, assert_matched_work
from benchmarks.quality.objectives import MIXED_3D, SPHERE_2D
from benchmarks.quality.runner import (
    RunSettings,
    Suggested,
    _capture_upstream_numerical_messages,
    _UpstreamNumericalDiagnostics,
    make_adapter,
    run_trial,
)


def test_work_budget_fails_closed_on_unknown_or_changed_work() -> None:
    baseline = WorkBudget(
        objective_evaluations=20,
        batch_size=2,
        population_size=100,
        generations=100,
        gp_initial_steps=100,
        gp_update_steps=100,
    )
    assert_matched_work(baseline, baseline)
    changed = WorkBudget(
        objective_evaluations=20,
        batch_size=2,
        population_size=100,
        generations=10,
        gp_initial_steps=100,
        gp_update_steps=None,
    )
    with pytest.raises(ValueError, match=r"generations.*gp_update_steps"):
        assert_matched_work(baseline, changed)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"objective_evaluations": 1.5}, "objective_evaluations"),
        ({"batch_size": True}, "batch_size"),
        ({"population_size": 3.5}, "population_size"),
        ({"generations": False}, "generations"),
    ],
)
def test_work_budget_rejects_non_integer_counts(overrides: dict[str, object], field: str) -> None:
    values: dict[str, object] = {"objective_evaluations": 8, "batch_size": 2}
    values.update(overrides)
    with pytest.raises(TypeError, match=field):
        WorkBudget(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ("objective_evaluations", "batch_size"))
def test_work_budget_rejects_missing_required_counts(field: str) -> None:
    values: dict[str, object] = {"objective_evaluations": 8, "batch_size": 2}
    values[field] = None
    with pytest.raises(TypeError, match=field):
        WorkBudget(**values)  # type: ignore[arg-type]


def test_work_budget_normalizes_and_validates_search_candidate_work() -> None:
    work = WorkBudget(
        objective_evaluations=8,
        batch_size=2,
        population_size=100,
        generations=99,
        search_candidate_evaluations=10_000,
    )
    assert work.expected_search_evaluations == 10_000
    with pytest.raises(ValueError, match="population_size"):
        WorkBudget(
            objective_evaluations=8,
            batch_size=2,
            population_size=100,
            generations=99,
            search_candidate_evaluations=10_100,
        )


def test_work_budget_loader_rejects_missing_and_unknown_fields() -> None:
    raw = WorkBudget(objective_evaluations=8, batch_size=2).to_dict()
    missing = dict(raw)
    missing.pop("dtype")
    with pytest.raises(ValueError, match=r"missing fields.*dtype"):
        WorkBudget.from_dict(missing)

    extra: dict[str, object] = {**raw, "legacy_mode": True}
    with pytest.raises(ValueError, match=r"unknown fields.*legacy_mode"):
        WorkBudget.from_dict(extra)


def test_phase_recorder_and_result_writer_emit_finite_json(tmp_path: Path) -> None:
    recorder = PhaseRecorder()
    with recorder.phase("suggest.total"):
        sum(range(10))
    result = BenchmarkResult(
        implementation={"name": "test-double"},
        suite="unit",
        case="serialization",
        seed=3,
        work=WorkBudget(objective_evaluations=1, batch_size=1),
        phases=recorder.to_dict(),
    )
    destination = write_result(result, tmp_path / "raw.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["work"]["objective_evaluations"] == 1
    assert payload["phases"]["suggest.total"]["wall_seconds"][0] >= 0
    assert payload["runtime"]["packages"]["torch"] is not None
    with pytest.raises(FileExistsError):
        write_result(result, destination)


def test_pinned_upstream_manifest_uses_full_commit_and_is_development_only() -> None:
    path = Path(__file__).resolve().parents[1] / "environments" / "upstream-hebo.json"
    manifest = UpstreamManifest.load(path)
    assert manifest.commit == "ee6112d39d1a9e9703fecaf9057193e1ec9dae72"
    assert manifest.development_only
    lock_path = path.parent / manifest.dependency_lock
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == manifest.dependency_lock_sha256


def test_toy_objectives_have_exact_known_optima() -> None:
    assert SPHERE_2D.evaluate([{"x0": 0.0, "x1": 0.0}]) == [0.0]
    assert MIXED_3D.evaluate([{"x": 1.25, "depth": 3, "kind": "b"}]) == [0.0]
    assert MIXED_3D.normalized_regret(MIXED_3D.regret_scale) == 1.0


def test_upstream_numerical_messages_are_counted_and_preserved() -> None:
    gp_module = ModuleType("fake_upstream_gp")
    forwarded: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_print(*values: object, **options: object) -> None:
        forwarded.append((values, options))

    gp_module.print = original_print  # type: ignore[attr-defined]
    diagnostics = _UpstreamNumericalDiagnostics()
    with _capture_upstream_numerical_messages(gp_module, diagnostics):
        gp_module.print("jitter = 1e-07", flush=True)
        gp_module.print("jitter = 1e-06")
        gp_module.print("jitter is too large, give up fitting GP")
        gp_module.print("jitter is too large, output random predictions")
        gp_module.print("unrelated upstream message")

    assert gp_module.print is original_print
    assert forwarded == [
        (("jitter = 1e-07",), {"flush": True}),
        (("jitter = 1e-06",), {}),
        (("jitter is too large, give up fitting GP",), {}),
        (("jitter is too large, output random predictions",), {}),
        (("unrelated upstream message",), {}),
    ]
    assert diagnostics.to_dict() == {
        "jitter": {"escalations": 2, "maximum_parameter": 1e-6},
        "fit": {"give_ups": 1},
        "prediction": {"random_fallbacks": 1},
    }


def test_upstream_numerical_capture_restores_missing_print_after_failure() -> None:
    gp_module = ModuleType("fake_upstream_gp")
    diagnostics = _UpstreamNumericalDiagnostics()

    with (
        pytest.raises(RuntimeError, match="boom"),
        _capture_upstream_numerical_messages(gp_module, diagnostics),
    ):
        raise RuntimeError("boom")

    assert not hasattr(gp_module, "print")


def test_leanhebo_quality_smoke_uses_public_suggest_observe_contract() -> None:
    settings = RunSettings(
        evaluation_budget=2,
        batch_size=2,
        random_samples=4,
        population_size=4,
        generations=0,
        gp_initial_steps=0,
        gp_update_steps=0,
    )
    result = run_trial(make_adapter("leanhebo", SPHERE_2D, settings, seed=7), SPHERE_2D, 7)
    payload = result.to_dict()
    assert not result.failures
    metrics = payload["metrics"]
    quality = payload["quality"]
    assert isinstance(metrics, dict)
    assert isinstance(quality, dict)
    assert metrics["evaluations_completed"] == 2
    assert quality["normalized_regret"] is not None
    implementation_metrics = metrics["implementation_metrics"]
    assert isinstance(implementation_metrics, dict)
    assert implementation_metrics["numerical_stability"] == {
        "jitter": {"escalations": 0, "maximum_parameter": None},
        "fit": {"give_ups": 0},
        "prediction": {"random_fallbacks": 0},
    }
    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    assert len(str(implementation["commit"])) == 40
    assert isinstance(implementation["source_dirty"], bool)


def test_lean_adapter_declares_cold_and_persistent_lifecycle_work() -> None:
    cold = make_adapter(
        "leanhebo",
        SPHERE_2D,
        RunSettings(
            evaluation_budget=4,
            batch_size=2,
            random_samples=2,
            population_size=4,
            generations=0,
            gp_initial_steps=3,
            gp_update_steps=1,
            posterior_batch_size=None,
            gp_optimizer="psgld",
            model_lifecycle="cold",
        ),
        seed=1,
    ).work
    persistent = make_adapter(
        "leanhebo",
        SPHERE_2D,
        RunSettings(
            evaluation_budget=4,
            batch_size=2,
            random_samples=2,
            population_size=4,
            generations=0,
            gp_initial_steps=3,
            gp_update_steps=1,
            posterior_batch_size=None,
            gp_optimizer="psgld",
            model_lifecycle="persistent",
        ),
        seed=1,
    ).work

    assert cold.gp_update_steps == 3
    assert cold.full_refit_interval == 1
    assert cold.reuse_parameters is False
    assert cold.reuse_optimizer_state is False
    assert cold.use_set_train_data is False
    assert persistent.gp_update_steps == 1
    assert persistent.full_refit_interval is None
    assert persistent.reuse_parameters is True
    assert persistent.reuse_optimizer_state is True
    assert persistent.use_set_train_data is True
    assert cold.posterior_batch_size is None
    assert cold.gp_optimizer == persistent.gp_optimizer == "psgld"


def test_upstream_adapter_rejects_settings_it_cannot_apply_before_importing_upstream() -> None:
    settings = RunSettings(
        population_size=12,
        generations=2,
        posterior_batch_size=64,
        model_lifecycle="persistent",
    )

    with pytest.raises(ValueError) as error:
        make_adapter("upstream-hebo", SPHERE_2D, settings, seed=1)

    message = str(error.value)
    assert "model_lifecycle='cold'" in message
    assert "population_size=100" in message
    assert "generations=99" in message
    assert "posterior_batch_size=None" in message


class _SequenceAdapter:
    implementation: ClassVar[dict[str, object]] = {"name": "sequence-double"}
    work = WorkBudget(objective_evaluations=5, batch_size=1, random_samples=2)

    def __init__(self) -> None:
        self._index = 0

    def suggest(self, count: int) -> Suggested:
        assert count == 1
        values = (4.0, 3.0, 2.0, 2.0, 1.0)
        value = values[self._index]
        self._index += 1
        return Suggested([{"x0": value, "x1": 0.0}], native=None)

    def observe(self, suggested: Suggested, values: Sequence[float]) -> None:
        assert len(suggested.rows) == len(values) == 1

    def phase_wall_times(self) -> dict[str, list[float]]:
        return {}

    def metrics(self) -> dict[str, object]:
        return {}


def test_trial_separates_sobol_first_model_and_steady_model_phases() -> None:
    result = run_trial(_SequenceAdapter(), SPHERE_2D, seed=4)

    assert len(result.phases["driver.suggest.initial_sobol"]["wall_seconds"]) == 2
    assert len(result.phases["driver.suggest.first_model"]["wall_seconds"]) == 1
    assert len(result.phases["driver.suggest.steady_model"]["wall_seconds"]) == 2
    assert result.metrics["duplicate_suggestions"] == 1


class _MisreportedSearchAdapter(_SequenceAdapter):
    work = WorkBudget(
        objective_evaluations=3,
        batch_size=1,
        random_samples=2,
        population_size=2,
        generations=0,
        search_candidate_evaluations=2,
    )

    def suggest(self, count: int) -> Suggested:
        suggested = super().suggest(count)
        if self._index <= 2:
            return suggested
        return Suggested(
            suggested.rows,
            suggested.native,
            {"objective_calls": 1, "candidate_evaluations": 1, "offspring_generations": 0},
        )


def test_trial_fails_when_actual_search_work_differs_from_declaration() -> None:
    result = run_trial(_MisreportedSearchAdapter(), SPHERE_2D, seed=4)

    assert result.metrics["evaluations_completed"] == 2
    assert result.failures[0]["stage"] == "search-work"
    assert "actual=1, declared=2" in str(result.failures[0]["message"])
