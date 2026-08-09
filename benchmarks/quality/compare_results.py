# SPDX-License-Identifier: MIT

"""Compare paired raw benchmark results without hiding changed algorithmic work."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from benchmarks.harness.work import WorkBudget, assert_matched_work

PairKey: TypeAlias = tuple[str, str, int]
RawResult: TypeAlias = dict[str, Any]

_SUGGEST_PHASES = (
    "driver.suggest.initial_sobol",
    "driver.suggest.first_model",
    "driver.suggest.steady_model",
)
_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_SEED = 17_291


@dataclass(frozen=True, slots=True)
class _NumericalStability:
    jitter_escalations: int
    maximum_jitter_parameter: float | None
    fit_give_ups: int
    random_prediction_fallbacks: int

    @property
    def degraded(self) -> bool:
        return self.fit_give_ups > 0 or self.random_prediction_fallbacks > 0


def load_result_set(source: str | Path) -> dict[PairKey, RawResult]:
    """Load one file or a directory tree of version-current raw result JSON files."""

    path = Path(source)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.rglob("*.json"))
    else:
        raise FileNotFoundError(f"benchmark result source does not exist: {path}")
    if not paths:
        raise ValueError(f"no JSON benchmark results found beneath {path}")

    results: dict[PairKey, RawResult] = {}
    for result_path in paths:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"raw benchmark result is not an object: {result_path}")
        benchmark = _mapping(raw.get("benchmark"), path=f"{result_path}:benchmark")
        suite = _required_string(benchmark.get("suite"), path="benchmark.suite")
        case = _required_string(benchmark.get("case"), path="benchmark.case")
        seed = benchmark.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{result_path}: benchmark.seed must be an integer")
        key = (suite, case, seed)
        if key in results:
            raise ValueError(f"duplicate benchmark result for {key!r} beneath {path}")
        results[key] = raw
    return results


def build_comparison_report(
    candidate_results: Mapping[PairKey, RawResult],
    baseline_results: Mapping[PairKey, RawResult],
    *,
    allow_changed_work: bool = False,
) -> dict[str, object]:
    """Build a paired report, failing closed on changed work by default."""

    candidate_keys = set(candidate_results)
    baseline_keys = set(baseline_results)
    if candidate_keys != baseline_keys:
        missing_candidate = sorted(baseline_keys - candidate_keys)
        missing_baseline = sorted(candidate_keys - baseline_keys)
        raise ValueError(
            "benchmark seeds are not paired "
            f"(missing candidate={missing_candidate}, missing baseline={missing_baseline})"
        )
    if not candidate_keys:
        raise ValueError("cannot compare empty benchmark result sets")

    work_mismatches: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[PairKey]] = defaultdict(list)
    for key in sorted(candidate_keys):
        candidate_work = _work(candidate_results[key])
        baseline_work = _work(baseline_results[key])
        try:
            assert_matched_work(candidate_work, baseline_work)
        except ValueError as error:
            mismatch = {
                "suite": key[0],
                "case": key[1],
                "seed": key[2],
                "message": str(error),
                "differences": _work_differences(candidate_work, baseline_work),
            }
            work_mismatches.append(mismatch)
            if not allow_changed_work:
                raise ValueError(
                    f"refusing changed-work comparison for {key!r}: {error}"
                ) from error
        grouped[(key[0], key[1])].append(key)

    cases: dict[str, object] = {}
    for (suite, case), keys in sorted(grouped.items()):
        candidate_group = [candidate_results[key] for key in keys]
        baseline_group = [baseline_results[key] for key in keys]
        eligible_pairs = [
            (candidate, baseline)
            for candidate, baseline in zip(candidate_group, baseline_group, strict=True)
            if _eligible_for_estimates(candidate) and _eligible_for_estimates(baseline)
        ]
        eligible_candidates = [candidate for candidate, _ in eligible_pairs]
        eligible_baselines = [baseline for _, baseline in eligible_pairs]
        candidate_summary = _summarize_records(
            candidate_group,
            estimate_records=eligible_candidates,
        )
        baseline_summary = _summarize_records(
            baseline_group,
            estimate_records=eligible_baselines,
        )
        cases[f"{suite}/{case}"] = {
            "paired_seeds": [key[2] for key in keys],
            "pair_counts": {
                "total": len(keys),
                "eligible": len(eligible_pairs),
                "excluded": len(keys) - len(eligible_pairs),
            },
            "candidate": candidate_summary,
            "baseline": baseline_summary,
            "baseline_over_candidate_speedup": {
                phase: _paired_phase_ratio(eligible_candidates, eligible_baselines, phase)
                for phase in _SUGGEST_PHASES
            },
            "baseline_minus_candidate_regret": _paired_regret_difference(
                eligible_candidates, eligible_baselines
            ),
        }

    return {
        "comparison_lane": "changed-work" if work_mismatches else "matched-work",
        "allow_changed_work": allow_changed_work,
        "candidate": _implementation_identity(candidate_results.values()),
        "baseline": _implementation_identity(baseline_results.values()),
        "paired_result_count": len(candidate_keys),
        "work_mismatches": work_mismatches,
        "cases": cases,
    }


def render_report(report: Mapping[str, object]) -> str:
    """Render a compact human-readable report with an unambiguous lane label."""

    lane = _required_string(report.get("comparison_lane"), path="comparison_lane")
    candidate = _mapping(report.get("candidate"), path="candidate")
    baseline = _mapping(report.get("baseline"), path="baseline")
    lines = [
        f"Comparison lane: {lane.upper()}",
        f"Candidate: {_required_string(candidate.get('name'), path='candidate.name')}",
        f"Baseline: {_required_string(baseline.get('name'), path='baseline.name')}",
        f"Paired results: {_display(report.get('paired_result_count', 0))}",
    ]
    if lane == "changed-work":
        lines.append("WARNING: speed ratios below are changed-work observations, not speedups.")

    cases = _mapping(report.get("cases"), path="cases")
    for label, raw_case in sorted(cases.items()):
        case = _mapping(raw_case, path=f"cases.{label}")
        lines.append("")
        lines.append(f"[{label}]")
        pair_counts = _mapping(case.get("pair_counts"), path=f"cases.{label}.pair_counts")
        lines.append(
            "Estimate pairs: "
            f"total={pair_counts.get('total', 0)}, "
            f"eligible={pair_counts.get('eligible', 0)}, "
            f"excluded={pair_counts.get('excluded', 0)}"
        )
        lines.append(
            "Exclusion rule: omit pairs with raw failures or numerical fit/prediction "
            "fallbacks from phase and regret estimates; jitter-only pairs remain eligible."
        )
        for side in ("candidate", "baseline"):
            summary = _mapping(case.get(side), path=f"cases.{label}.{side}")
            lines.append(
                f"{side}: failures={summary.get('failures', 0)}, "
                f"duplicates={summary.get('duplicate_suggestions', 0)}, "
                f"median_regret={_display(summary.get('median_normalized_regret'))}"
            )
            numerical = _mapping(
                summary.get("numerical_stability"),
                path=f"cases.{label}.{side}.numerical_stability",
            )
            if numerical.get("reported_trials", 0):
                lines.append(
                    f"{side} numerical stability: "
                    f"degraded_trials={numerical.get('degraded_trials', 0)}, "
                    f"jitter_escalations={numerical.get('jitter_escalations', 0)}, "
                    f"fit_give_ups={numerical.get('fit_give_ups', 0)}, "
                    "random_prediction_fallbacks="
                    f"{numerical.get('random_prediction_fallbacks', 0)}"
                )
        speedups = _mapping(
            case.get("baseline_over_candidate_speedup"),
            path=f"cases.{label}.baseline_over_candidate_speedup",
        )
        for phase in _SUGGEST_PHASES:
            summary = _mapping(speedups.get(phase), path=f"speedups.{phase}")
            lines.append(
                f"{phase}: {_display(summary.get('estimate'))}x baseline/candidate "
                f"(paired bootstrap 95% CI {_display_interval(summary.get('ci95'))})"
            )
        regret = _mapping(
            case.get("baseline_minus_candidate_regret"),
            path=f"cases.{label}.baseline_minus_candidate_regret",
        )
        lines.append(
            "baseline-candidate regret: "
            f"{_display(regret.get('estimate'))} "
            f"(paired bootstrap 95% CI {_display_interval(regret.get('ci95'))})"
        )
    return "\n".join(lines)


def _summarize_records(
    records: Sequence[RawResult],
    *,
    estimate_records: Sequence[RawResult],
) -> dict[str, object]:
    regrets: list[float] = []
    failures = 0
    duplicates = 0
    phase_samples: dict[str, list[float]] = {name: [] for name in _SUGGEST_PHASES}
    for result in records:
        failures += int(_has_raw_failure(result))
        metrics = _mapping(result.get("metrics"), path="metrics")
        duplicate_value = metrics.get("duplicate_suggestions", 0)
        if isinstance(duplicate_value, bool) or not isinstance(duplicate_value, int):
            raise TypeError("metrics.duplicate_suggestions must be an integer")
        duplicates += duplicate_value

    for result in estimate_records:
        quality = _mapping(result.get("quality"), path="quality")
        regret = quality.get("normalized_regret")
        if regret is not None:
            numeric = _finite_number(regret, path="quality.normalized_regret")
            regrets.append(numeric)
        phases = _mapping(result.get("phases"), path="phases")
        for phase_name in _SUGGEST_PHASES:
            phase = phases.get(phase_name)
            if phase is None:
                continue
            phase_map = _mapping(phase, path=f"phases.{phase_name}")
            samples = phase_map.get("wall_seconds", [])
            if not isinstance(samples, list):
                raise TypeError(f"phases.{phase_name}.wall_seconds must be an array")
            phase_samples[phase_name].extend(
                _finite_number(sample, path=f"phases.{phase_name}.wall_seconds")
                for sample in samples
            )

    return {
        "trials": len(records),
        "failures": failures,
        "duplicate_suggestions": duplicates,
        "median_normalized_regret": None if not regrets else statistics.median(regrets),
        "mean_normalized_regret": None if not regrets else statistics.fmean(regrets),
        "phases": {name: _sample_summary(samples) for name, samples in phase_samples.items()},
        "numerical_stability": _summarize_numerical_stability(records),
    }


def _summarize_numerical_stability(records: Sequence[RawResult]) -> dict[str, object]:
    reported_trials = 0
    jitter_escalations = 0
    maximum_jitter_parameter: float | None = None
    fit_give_ups = 0
    fit_give_up_trials = 0
    random_prediction_fallbacks = 0
    random_prediction_fallback_trials = 0
    degraded_trials = 0
    for result in records:
        numerical = _record_numerical_stability(result)
        if numerical is None:
            continue
        jitter_escalations += numerical.jitter_escalations
        fit_give_ups += numerical.fit_give_ups
        random_prediction_fallbacks += numerical.random_prediction_fallbacks
        fit_give_up_trials += int(numerical.fit_give_ups > 0)
        random_prediction_fallback_trials += int(numerical.random_prediction_fallbacks > 0)
        degraded_trials += int(numerical.degraded)
        maximum = numerical.maximum_jitter_parameter
        if maximum is not None:
            maximum_jitter_parameter = (
                maximum
                if maximum_jitter_parameter is None
                else max(maximum_jitter_parameter, maximum)
            )
        reported_trials += 1
    return {
        "reported_trials": reported_trials,
        "jitter_escalations": jitter_escalations,
        "maximum_jitter_parameter": maximum_jitter_parameter,
        "fit_give_ups": fit_give_ups,
        "fit_give_up_trials": fit_give_up_trials,
        "random_prediction_fallbacks": random_prediction_fallbacks,
        "random_prediction_fallback_trials": random_prediction_fallback_trials,
        "degraded_trials": degraded_trials,
    }


def _eligible_for_estimates(result: RawResult) -> bool:
    if _has_raw_failure(result):
        return False
    numerical = _record_numerical_stability(result)
    return numerical is None or not numerical.degraded


def _has_raw_failure(result: RawResult) -> bool:
    raw_failures = result.get("failures", [])
    if not isinstance(raw_failures, list):
        raise TypeError("result.failures must be an array")
    return bool(raw_failures)


def _record_numerical_stability(
    result: RawResult,
) -> _NumericalStability | None:
    metrics = _mapping(result.get("metrics"), path="metrics")
    implementation_metrics = metrics.get("implementation_metrics")
    if implementation_metrics is None:
        return None
    implementation_map = _mapping(
        implementation_metrics,
        path="metrics.implementation_metrics",
    )
    raw_numerical = implementation_map.get("numerical_stability")
    if raw_numerical is None:
        return None
    numerical = _mapping(
        raw_numerical,
        path="metrics.implementation_metrics.numerical_stability",
    )
    jitter = _mapping(numerical.get("jitter"), path="numerical_stability.jitter")
    fit = _mapping(numerical.get("fit"), path="numerical_stability.fit")
    prediction = _mapping(
        numerical.get("prediction"),
        path="numerical_stability.prediction",
    )
    escalations = _non_negative_int(
        jitter.get("escalations"),
        path="numerical_stability.jitter.escalations",
    )
    give_ups = _non_negative_int(
        fit.get("give_ups"),
        path="numerical_stability.fit.give_ups",
    )
    random_fallbacks = _non_negative_int(
        prediction.get("random_fallbacks"),
        path="numerical_stability.prediction.random_fallbacks",
    )
    raw_maximum = jitter.get("maximum_parameter")
    maximum = (
        None
        if raw_maximum is None
        else _finite_number(
            raw_maximum,
            path="numerical_stability.jitter.maximum_parameter",
        )
    )
    if maximum is not None and maximum < 0:
        raise ValueError("numerical_stability.jitter.maximum_parameter must be non-negative")
    return _NumericalStability(
        jitter_escalations=escalations,
        maximum_jitter_parameter=maximum,
        fit_give_ups=give_ups,
        random_prediction_fallbacks=random_fallbacks,
    )


def _paired_phase_ratio(
    candidate_records: Sequence[RawResult],
    baseline_records: Sequence[RawResult],
    phase: str,
) -> dict[str, object]:
    ratios: list[float] = []
    for candidate, baseline in zip(candidate_records, baseline_records, strict=True):
        candidate_median = _record_phase_median(candidate, phase)
        baseline_median = _record_phase_median(baseline, phase)
        if candidate_median is None or baseline_median is None or candidate_median <= 0:
            continue
        ratios.append(baseline_median / candidate_median)
    return _paired_bootstrap_summary(ratios)


def _paired_regret_difference(
    candidate_records: Sequence[RawResult], baseline_records: Sequence[RawResult]
) -> dict[str, object]:
    differences: list[float] = []
    for candidate, baseline in zip(candidate_records, baseline_records, strict=True):
        candidate_quality = _mapping(candidate.get("quality"), path="candidate.quality")
        baseline_quality = _mapping(baseline.get("quality"), path="baseline.quality")
        candidate_regret = candidate_quality.get("normalized_regret")
        baseline_regret = baseline_quality.get("normalized_regret")
        if candidate_regret is None or baseline_regret is None:
            continue
        differences.append(
            _finite_number(baseline_regret, path="baseline.quality.normalized_regret")
            - _finite_number(candidate_regret, path="candidate.quality.normalized_regret")
        )
    return _paired_bootstrap_summary(differences)


def _record_phase_median(result: RawResult, phase: str) -> float | None:
    phases = _mapping(result.get("phases"), path="phases")
    raw_phase = phases.get(phase)
    if raw_phase is None:
        return None
    phase_map = _mapping(raw_phase, path=f"phases.{phase}")
    samples = phase_map.get("wall_seconds", [])
    if not isinstance(samples, list):
        raise TypeError(f"phases.{phase}.wall_seconds must be an array")
    converted = [_finite_number(sample, path=f"phases.{phase}.wall_seconds") for sample in samples]
    return None if not converted else statistics.median(converted)


def _paired_bootstrap_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"pairs": 0, "estimate": None, "ci95": None}
    estimate = statistics.median(values)
    if len(values) == 1:
        return {"pairs": 1, "estimate": estimate, "ci95": [estimate, estimate]}
    generator = random.Random(_BOOTSTRAP_SEED)
    count = len(values)
    bootstrap = sorted(
        statistics.median([values[generator.randrange(count)] for _ in range(count)])
        for _ in range(_BOOTSTRAP_RESAMPLES)
    )
    lower = bootstrap[math.floor(0.025 * (_BOOTSTRAP_RESAMPLES - 1))]
    upper = bootstrap[math.ceil(0.975 * (_BOOTSTRAP_RESAMPLES - 1))]
    return {"pairs": count, "estimate": estimate, "ci95": [lower, upper]}


def _sample_summary(samples: Sequence[float]) -> dict[str, object]:
    if not samples:
        return {"samples": 0, "median_seconds": None, "mean_seconds": None}
    return {
        "samples": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
    }


def _implementation_identity(results: Iterable[RawResult]) -> dict[str, object]:
    identities: dict[str, dict[str, object]] = {}
    for result in results:
        implementation = _mapping(result.get("implementation"), path="implementation")
        name = _required_string(implementation.get("name"), path="implementation.name")
        identity = {
            "name": name,
            "version": implementation.get("version"),
            "commit": implementation.get("commit"),
            "source_dirty": implementation.get("source_dirty"),
        }
        identities[json.dumps(identity, sort_keys=True)] = identity
    if len(identities) != 1:
        raise ValueError("one result set contains multiple implementation identities")
    return next(iter(identities.values()))


def _work(result: Mapping[str, object]) -> WorkBudget:
    return WorkBudget.from_dict(_mapping(result.get("work"), path="work"))


def _work_differences(candidate: WorkBudget, baseline: WorkBudget) -> dict[str, dict[str, object]]:
    return {
        name: {"candidate": candidate_value, "baseline": baseline_value}
        for name, candidate_value in candidate.to_dict().items()
        if candidate_value != (baseline_value := baseline.to_dict()[name])
    }


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    return value


def _required_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite")
    return converted


def _non_negative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value < 0:
        raise ValueError(f"{path} must be non-negative")
    return value


def _display(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return str(value)


def _display_interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "n/a"
    return f"[{_display(value[0])}, {_display(value[1])}]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--allow-changed-work",
        action="store_true",
        help="report mismatched work with a warning instead of refusing the comparison",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_comparison_report(
        load_result_set(args.candidate),
        load_result_set(args.baseline),
        allow_changed_work=args.allow_changed_work,
    )
    print(render_report(report))
    if args.json_output is not None:
        if args.json_output.exists():
            raise FileExistsError(f"refusing to overwrite comparison report: {args.json_output}")
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
