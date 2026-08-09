# SPDX-License-Identifier: MIT

"""Compare fixed-history results, failing closed if the preloaded data differ."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import cast

from benchmarks.harness.work import WorkBudget
from benchmarks.latency.fixed_history import SUGGEST_PHASE, Family, FixedHistoryCell
from benchmarks.quality.compare_results import (
    PairKey,
    RawResult,
    build_comparison_report,
    load_result_set,
    render_report,
)


@dataclass(frozen=True, slots=True)
class FixedHistoryIdentity:
    history_sha256: str
    history_observations: int
    public_dimension: int
    family: Family
    repeats_requested: int


def build_fixed_history_report(
    candidate_results: Mapping[PairKey, RawResult],
    baseline_results: Mapping[PairKey, RawResult],
) -> dict[str, object]:
    """Build a strict matched-work report after checking fixed-history identity."""

    candidate_keys = set(candidate_results)
    baseline_keys = set(baseline_results)
    if candidate_keys != baseline_keys:
        # Delegate to the shared comparator for its complete missing-key diagnostic.
        return build_comparison_report(candidate_results, baseline_results)
    for key in sorted(candidate_keys):
        _validate_fixed_record(candidate_results[key], path=f"candidate[{key!r}]")
        _validate_fixed_record(baseline_results[key], path=f"baseline[{key!r}]")
        candidate_identity = _fixed_identity(candidate_results[key])
        baseline_identity = _fixed_identity(baseline_results[key])
        if candidate_identity != baseline_identity:
            raise ValueError(
                f"refusing fixed-history comparison for {key!r}: "
                f"candidate history={candidate_identity}, baseline history={baseline_identity}"
            )
    return build_comparison_report(candidate_results, baseline_results)


def _validate_fixed_record(result: Mapping[str, object], *, path: str) -> None:
    """Reject incomplete or invalid trials before computing timing ratios."""

    failures = result.get("failures")
    if not isinstance(failures, list):
        raise TypeError(f"{path}.failures must be an array")
    if failures:
        raise ValueError(f"{path}.failures must be empty")

    metrics = _mapping(result.get("metrics"), path=f"{path}.metrics")
    repeats_requested = _positive_int(
        metrics.get("repeats_requested"), path=f"{path}.metrics.repeats_requested"
    )
    repeats_completed = _non_negative_int(
        metrics.get("repeats_completed"), path=f"{path}.metrics.repeats_completed"
    )
    if repeats_completed != repeats_requested:
        raise ValueError(
            f"{path}.metrics.repeats_completed must equal repeats_requested ({repeats_requested})"
        )

    work = WorkBudget.from_dict(_mapping(result.get("work"), path=f"{path}.work"))
    suggestions_returned = _non_negative_int(
        metrics.get("suggestions_returned"), path=f"{path}.metrics.suggestions_returned"
    )
    expected_suggestions = repeats_requested * work.batch_size
    if suggestions_returned != expected_suggestions:
        raise ValueError(
            f"{path}.metrics.suggestions_returned must equal repeats_requested * batch_size "
            f"({expected_suggestions})"
        )
    for name in ("invalid_suggestions", "duplicate_suggestions"):
        value = _non_negative_int(metrics.get(name), path=f"{path}.metrics.{name}")
        if value:
            raise ValueError(f"{path}.metrics.{name} must be zero")

    phases = _mapping(result.get("phases"), path=f"{path}.phases")
    phase_path = f"{path}.phases.{SUGGEST_PHASE}"
    suggest_phase = _mapping(
        phases.get(SUGGEST_PHASE),
        path=phase_path,
    )
    for clock in ("wall_seconds", "process_cpu_seconds"):
        samples = suggest_phase.get(clock)
        if not isinstance(samples, list):
            raise TypeError(f"{phase_path}.{clock} must be an array")
        if len(samples) != repeats_requested:
            raise ValueError(f"{phase_path}.{clock} must contain {repeats_requested} samples")
        for index, sample in enumerate(samples):
            _non_negative_finite_number(
                sample,
                path=f"{phase_path}.{clock}[{index}]",
            )

    candidate_hashes = metrics.get("candidate_sha256")
    if not isinstance(candidate_hashes, list):
        raise TypeError(f"{path}.metrics.candidate_sha256 must be an array")
    if len(candidate_hashes) != repeats_requested:
        raise ValueError(
            f"{path}.metrics.candidate_sha256 must contain {repeats_requested} digests"
        )
    for index, digest in enumerate(candidate_hashes):
        _sha256(digest, path=f"{path}.metrics.candidate_sha256[{index}]")
    if len(set(candidate_hashes)) != 1:
        raise ValueError(f"{path}.metrics.candidate_sha256 contains nondeterministic digests")

    implementation_metrics = _mapping(
        metrics.get("implementation_metrics"),
        path=f"{path}.metrics.implementation_metrics",
    )
    repeat_metrics = implementation_metrics.get("repeats")
    if not isinstance(repeat_metrics, list):
        raise TypeError(f"{path}.metrics.implementation_metrics.repeats must be an array")
    if len(repeat_metrics) != repeats_requested:
        raise ValueError(
            f"{path}.metrics.implementation_metrics.repeats must contain "
            f"{repeats_requested} entries"
        )
    for index, raw_repeat in enumerate(repeat_metrics):
        repeat = _mapping(
            raw_repeat,
            path=f"{path}.metrics.implementation_metrics.repeats[{index}]",
        )
        numerical = repeat.get("numerical_stability")
        if numerical is not None:
            _reject_numerical_fallbacks(
                numerical,
                path=(
                    f"{path}.metrics.implementation_metrics.repeats[{index}].numerical_stability"
                ),
            )
    numerical = implementation_metrics.get("numerical_stability")
    if numerical is not None:
        _reject_numerical_fallbacks(
            numerical,
            path=f"{path}.metrics.implementation_metrics.numerical_stability",
        )


def _reject_numerical_fallbacks(value: object, *, path: str) -> None:
    numerical = _mapping(value, path=path)
    fit = _mapping(numerical.get("fit"), path=f"{path}.fit")
    prediction = _mapping(numerical.get("prediction"), path=f"{path}.prediction")
    give_ups = _non_negative_int(fit.get("give_ups"), path=f"{path}.fit.give_ups")
    random_fallbacks = _non_negative_int(
        prediction.get("random_fallbacks"),
        path=f"{path}.prediction.random_fallbacks",
    )
    if give_ups:
        raise ValueError(f"{path}.fit.give_ups must be zero")
    if random_fallbacks:
        raise ValueError(f"{path}.prediction.random_fallbacks must be zero")


def _fixed_identity(result: Mapping[str, object]) -> FixedHistoryIdentity:
    benchmark = _mapping(result.get("benchmark"), path="benchmark")
    suite = _string(benchmark.get("suite"), path="benchmark.suite")
    if suite != "fixed-history-latency":
        raise ValueError(f"expected fixed-history-latency result, got suite {suite!r}")
    case = _string(benchmark.get("case"), path="benchmark.case")
    metrics = _mapping(result.get("metrics"), path="metrics")
    history_sha256 = _string(metrics.get("history_sha256"), path="metrics.history_sha256")
    if len(history_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in history_sha256
    ):
        raise ValueError("metrics.history_sha256 must be a lowercase SHA-256 digest")
    history_observations = _positive_int(
        metrics.get("history_observations"), path="metrics.history_observations"
    )
    public_dimension = _positive_int(
        metrics.get("public_dimension"), path="metrics.public_dimension"
    )
    family = _string(metrics.get("family"), path="metrics.family")
    if family not in ("continuous", "mixed"):
        raise ValueError("metrics.family must be continuous or mixed")
    repeats_requested = _positive_int(
        metrics.get("repeats_requested"), path="metrics.repeats_requested"
    )
    work = WorkBudget.from_dict(_mapping(result.get("work"), path="work"))
    if work.objective_evaluations != history_observations:
        raise ValueError("fixed-history observation count differs from declared objective work")
    typed_family = cast(Family, family)
    expected_case = FixedHistoryCell(
        typed_family,
        public_dimension,
        history_observations,
        work.batch_size,
    ).name
    if case != expected_case:
        raise ValueError(f"fixed-history case label is {case!r}; expected {expected_case!r}")
    return FixedHistoryIdentity(
        history_sha256,
        history_observations,
        public_dimension,
        typed_family,
        repeats_requested,
    )


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    return value


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _positive_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{path} must be a positive integer")
    return value


def _non_negative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{path} must be a non-negative integer")
    return value


def _non_negative_finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{path} must be finite and non-negative")
    return converted


def _sha256(value: object, *, path: str) -> str:
    digest = _string(value, path=path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_fixed_history_report(
        load_result_set(args.candidate),
        load_result_set(args.baseline),
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
