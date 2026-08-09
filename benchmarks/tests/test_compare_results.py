# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from benchmarks.harness.results import BenchmarkResult
from benchmarks.harness.work import WorkBudget
from benchmarks.quality.compare_results import (
    PairKey,
    RawResult,
    build_comparison_report,
    render_report,
)


def _work() -> WorkBudget:
    return WorkBudget(
        objective_evaluations=8,
        batch_size=2,
        population_size=100,
        generations=99,
        gp_initial_steps=10,
        gp_update_steps=10,
        full_refit_interval=1,
        posterior_batch_size=None,
        random_samples=4,
        search_candidate_evaluations=10_000,
        gp_optimizer="psgld",
        learning_rate=0.01,
        reuse_parameters=False,
        reuse_optimizer_state=False,
        use_set_train_data=False,
    )


def _result(*, name: str, seed: int, seconds: float, work: WorkBudget, regret: float) -> RawResult:
    result = BenchmarkResult(
        implementation={"name": name, "version": "test", "commit": None},
        suite="toy-quality",
        case="sphere-2d",
        seed=seed,
        work=work,
        phases={
            "driver.suggest.initial_sobol": {
                "wall_seconds": [seconds / 10],
                "process_cpu_seconds": [],
            },
            "driver.suggest.first_model": {
                "wall_seconds": [seconds],
                "process_cpu_seconds": [],
            },
            "driver.suggest.steady_model": {
                "wall_seconds": [seconds * 0.8],
                "process_cpu_seconds": [],
            },
        },
        metrics={"duplicate_suggestions": seed},
        quality={"normalized_regret": regret},
    )
    return cast(RawResult, result.to_dict())


def test_matched_report_pairs_seeds_and_separates_suggestion_phases() -> None:
    work = _work()
    candidate: dict[PairKey, RawResult] = {
        ("toy-quality", "sphere-2d", seed): _result(
            name="leanhebo", seed=seed, seconds=float(seed + 1), work=work, regret=0.1
        )
        for seed in (0, 1)
    }
    baseline: dict[PairKey, RawResult] = {
        ("toy-quality", "sphere-2d", seed): _result(
            name="upstream-hebo",
            seed=seed,
            seconds=float(2 * (seed + 1)),
            work=work,
            regret=0.2,
        )
        for seed in (0, 1)
    }

    report = build_comparison_report(candidate, baseline)

    assert report["comparison_lane"] == "matched-work"
    case = cast(
        dict[str, object], cast(dict[str, object], report["cases"])["toy-quality/sphere-2d"]
    )
    speedups = cast(dict[str, dict[str, object]], case["baseline_over_candidate_speedup"])
    first_model = speedups["driver.suggest.first_model"]
    assert first_model["estimate"] == pytest.approx(2.0)
    assert first_model["ci95"] == pytest.approx([2.0, 2.0])
    regret = cast(dict[str, object], case["baseline_minus_candidate_regret"])
    assert regret["estimate"] == pytest.approx(0.1)
    assert regret["ci95"] == pytest.approx([0.1, 0.1])
    candidate_summary = cast(dict[str, object], case["candidate"])
    assert candidate_summary["duplicate_suggestions"] == 1
    rendered = render_report(report)
    assert "Comparison lane: MATCHED-WORK" in rendered
    assert "paired bootstrap 95% CI [2, 2]" in rendered


def test_changed_work_is_rejected_unless_explicitly_labeled() -> None:
    candidate_work = replace(
        _work(),
        gp_update_steps=2,
        full_refit_interval=None,
        reuse_parameters=True,
        reuse_optimizer_state=True,
        use_set_train_data=True,
    )
    key = ("toy-quality", "sphere-2d", 0)
    candidate = {
        key: _result(name="leanhebo", seed=0, seconds=1.0, work=candidate_work, regret=0.1)
    }
    baseline = {key: _result(name="upstream-hebo", seed=0, seconds=2.0, work=_work(), regret=0.2)}

    with pytest.raises(ValueError, match="refusing changed-work comparison"):
        build_comparison_report(candidate, baseline)

    report = build_comparison_report(candidate, baseline, allow_changed_work=True)
    assert report["comparison_lane"] == "changed-work"
    mismatches = cast(list[dict[str, object]], report["work_mismatches"])
    differences = cast(dict[str, object], mismatches[0]["differences"])
    assert "gp_update_steps" in differences
    assert "reuse_parameters" in differences
    rendered = render_report(report)
    assert "Comparison lane: CHANGED-WORK" in rendered
    assert "not speedups" in rendered
