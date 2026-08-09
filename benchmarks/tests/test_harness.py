# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.environments.prepare_upstream import UpstreamManifest
from benchmarks.harness.results import BenchmarkResult, PhaseRecorder, write_result
from benchmarks.harness.work import WorkBudget, assert_matched_work
from benchmarks.quality.objectives import MIXED_3D, SPHERE_2D
from benchmarks.quality.runner import RunSettings, make_adapter, run_trial


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


def test_phase_recorder_and_result_writer_emit_finite_versioned_json(tmp_path: Path) -> None:
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
    assert payload["schema_version"] == 1
    assert payload["work"]["objective_evaluations"] == 1
    assert payload["phases"]["suggest.total"]["wall_seconds"][0] >= 0
    with pytest.raises(FileExistsError):
        write_result(result, destination)


def test_pinned_upstream_manifest_uses_full_commit_and_is_development_only() -> None:
    path = Path(__file__).resolve().parents[1] / "environments" / "upstream-hebo.json"
    manifest = UpstreamManifest.load(path)
    assert manifest.commit == "ee6112d39d1a9e9703fecaf9057193e1ec9dae72"
    assert manifest.development_only


def test_toy_objectives_have_exact_known_optima() -> None:
    assert SPHERE_2D.evaluate([{"x0": 0.0, "x1": 0.0}]) == [0.0]
    assert MIXED_3D.evaluate([{"x": 1.25, "depth": 3, "kind": "b"}]) == [0.0]
    assert MIXED_3D.normalized_regret(MIXED_3D.regret_scale) == 1.0


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
    assert result.failures == () or not result.failures
    metrics = payload["metrics"]
    quality = payload["quality"]
    assert isinstance(metrics, dict)
    assert isinstance(quality, dict)
    assert metrics["evaluations_completed"] == 2
    assert quality["normalized_regret"] is not None
