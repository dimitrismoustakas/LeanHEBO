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


def _with_numerical_metrics(
    result: RawResult,
    *,
    escalations: int,
    maximum: float | None,
    give_ups: int,
    random_fallbacks: int,
) -> RawResult:
    result["metrics"]["implementation_metrics"] = {
        "numerical_stability": {
            "jitter": {"escalations": escalations, "maximum_parameter": maximum},
            "fit": {"give_ups": give_ups},
            "prediction": {"random_fallbacks": random_fallbacks},
        }
    }
    return result


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
    assert case["pair_counts"] == {"total": 2, "eligible": 2, "excluded": 0}
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


def test_report_aggregates_and_displays_upstream_numerical_fallbacks() -> None:
    key_zero = ("toy-quality", "sphere-2d", 0)
    key_one = ("toy-quality", "sphere-2d", 1)
    candidate = {
        key: _result(name="leanhebo", seed=key[2], seconds=1.0, work=_work(), regret=0.1)
        for key in (key_zero, key_one)
    }
    baseline = {
        key_zero: _with_numerical_metrics(
            _result(name="upstream-hebo", seed=0, seconds=2.0, work=_work(), regret=0.2),
            escalations=2,
            maximum=1e-6,
            give_ups=1,
            random_fallbacks=0,
        ),
        key_one: _with_numerical_metrics(
            _result(name="upstream-hebo", seed=1, seconds=2.0, work=_work(), regret=0.2),
            escalations=3,
            maximum=1e-5,
            give_ups=0,
            random_fallbacks=4,
        ),
    }

    report = build_comparison_report(candidate, baseline)

    case = cast(
        dict[str, object], cast(dict[str, object], report["cases"])["toy-quality/sphere-2d"]
    )
    baseline_summary = cast(dict[str, object], case["baseline"])
    numerical = cast(dict[str, object], baseline_summary["numerical_stability"])
    assert numerical == {
        "reported_trials": 2,
        "jitter_escalations": 5,
        "maximum_jitter_parameter": 1e-5,
        "fit_give_ups": 1,
        "fit_give_up_trials": 1,
        "random_prediction_fallbacks": 4,
        "random_prediction_fallback_trials": 1,
        "degraded_trials": 2,
    }
    rendered = render_report(report)
    assert "baseline numerical stability: degraded_trials=2, jitter_escalations=5" in rendered
    assert "fit_give_ups=1, random_prediction_fallbacks=4" in rendered


def test_estimates_exclude_failed_or_degraded_pairs_but_keep_jitter_only_pairs() -> None:
    keys = [("toy-quality", "sphere-2d", seed) for seed in range(5)]
    candidate_seconds = [1.0, 2.0, 10.0, 20.0, 30.0]
    baseline_seconds = [2.0, 6.0, 100.0, 200.0, 300.0]
    candidate = {
        key: _result(
            name="leanhebo",
            seed=key[2],
            seconds=candidate_seconds[key[2]],
            work=_work(),
            regret=[0.1, 0.1, 5.0, 6.0, 7.0][key[2]],
        )
        for key in keys
    }
    baseline = {
        key: _result(
            name="upstream-hebo",
            seed=key[2],
            seconds=baseline_seconds[key[2]],
            work=_work(),
            regret=[0.2, 0.5, 10.0, 20.0, 30.0][key[2]],
        )
        for key in keys
    }
    _with_numerical_metrics(
        baseline[keys[1]],
        escalations=3,
        maximum=1e-5,
        give_ups=0,
        random_fallbacks=0,
    )
    _with_numerical_metrics(
        baseline[keys[2]],
        escalations=4,
        maximum=1e-4,
        give_ups=1,
        random_fallbacks=0,
    )
    _with_numerical_metrics(
        candidate[keys[3]],
        escalations=0,
        maximum=None,
        give_ups=0,
        random_fallbacks=2,
    )
    baseline[keys[4]]["failures"] = [{"phase": "driver.suggest.first_model"}]

    report = build_comparison_report(candidate, baseline)

    case = cast(
        dict[str, object], cast(dict[str, object], report["cases"])["toy-quality/sphere-2d"]
    )
    assert case["pair_counts"] == {"total": 5, "eligible": 2, "excluded": 3}
    speedups = cast(dict[str, dict[str, object]], case["baseline_over_candidate_speedup"])
    assert speedups["driver.suggest.first_model"]["pairs"] == 2
    assert speedups["driver.suggest.first_model"]["estimate"] == pytest.approx(2.5)
    regret = cast(dict[str, object], case["baseline_minus_candidate_regret"])
    assert regret["pairs"] == 2
    assert regret["estimate"] == pytest.approx(0.25)

    candidate_summary = cast(dict[str, object], case["candidate"])
    baseline_summary = cast(dict[str, object], case["baseline"])
    assert candidate_summary["median_normalized_regret"] == pytest.approx(0.1)
    assert baseline_summary["median_normalized_regret"] == pytest.approx(0.35)
    candidate_phases = cast(dict[str, dict[str, object]], candidate_summary["phases"])
    baseline_phases = cast(dict[str, dict[str, object]], baseline_summary["phases"])
    assert candidate_phases["driver.suggest.first_model"]["median_seconds"] == pytest.approx(1.5)
    assert baseline_phases["driver.suggest.first_model"]["median_seconds"] == pytest.approx(4.0)
    assert cast(dict[str, object], candidate_summary["numerical_stability"])["degraded_trials"] == 1
    assert cast(dict[str, object], baseline_summary["numerical_stability"])["degraded_trials"] == 1

    rendered = render_report(report)
    assert "Estimate pairs: total=5, eligible=2, excluded=3" in rendered
    assert "jitter-only pairs remain eligible" in rendered
