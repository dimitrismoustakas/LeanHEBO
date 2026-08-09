# SPDX-License-Identifier: MIT

"""Fixed-history suggestion latency with an auditable, matched work contract."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from benchmarks.harness.results import BenchmarkResult, PhaseRecorder
from benchmarks.harness.work import WorkBudget, assert_matched_work
from benchmarks.quality.objectives import ParameterDefinition, ToyObjective
from benchmarks.quality.runner import RunSettings, Suggested, make_adapter

Family = Literal["continuous", "mixed"]

SUPPORTED_FAMILIES: tuple[Family, ...] = ("continuous", "mixed")
SUPPORTED_DIMENSIONS = (5, 20)
SUPPORTED_OBSERVATIONS = (16, 64, 128)
SUPPORTED_BATCH_SIZES = (1, 4)
SUGGEST_PHASE = "driver.suggest.first_model"


@dataclass(frozen=True, slots=True)
class FixedHistoryCell:
    """One cell in the declared fixed-history latency matrix."""

    family: Family
    dimension: int
    observations: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError(f"unsupported fixed-history family: {self.family!r}")
        for name, supported in (
            ("dimension", SUPPORTED_DIMENSIONS),
            ("observations", SUPPORTED_OBSERVATIONS),
            ("batch_size", SUPPORTED_BATCH_SIZES),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value not in supported:
                raise ValueError(f"{name} must be one of {supported}")

    @property
    def name(self) -> str:
        return f"{self.family}-d{self.dimension}-n{self.observations}-q{self.batch_size}"


@dataclass(frozen=True, slots=True)
class FixedHistory:
    """Deterministic observations shared byte-for-byte by both implementations."""

    rows: tuple[dict[str, object], ...]
    values: tuple[float, ...]
    sha256: str


def matrix_cells(
    families: Sequence[Family],
    dimensions: Sequence[int],
    observations: Sequence[int],
    batch_sizes: Sequence[int],
) -> tuple[FixedHistoryCell, ...]:
    """Expand selected axes in stable order, rejecting values outside the audited matrix."""

    _require_axis("families", families, SUPPORTED_FAMILIES)
    _require_axis("dimensions", dimensions, SUPPORTED_DIMENSIONS)
    _require_axis("observations", observations, SUPPORTED_OBSERVATIONS)
    _require_axis("batch sizes", batch_sizes, SUPPORTED_BATCH_SIZES)
    return tuple(
        FixedHistoryCell(family, dimension, observation_count, batch_size)
        for family in families
        for dimension in dimensions
        for observation_count in observations
        for batch_size in batch_sizes
    )


def make_objective(cell: FixedHistoryCell) -> ToyObjective:
    """Build the design schema used by a matrix cell.

    A mixed dimension follows a fixed float/integer/categorical rotation. Thus ``d`` always means
    the number of public parameters, rather than an implementation-specific encoded width.
    """

    parameters: list[ParameterDefinition] = []
    for index in range(cell.dimension):
        kind = (
            "float"
            if cell.family == "continuous"
            else ("float", "integer", "categorical")[index % 3]
        )
        if kind == "float":
            parameters.append(ParameterDefinition(f"x{index}", "float", -5.0, 5.0))
        elif kind == "integer":
            parameters.append(ParameterDefinition(f"i{index}", "integer", -5, 5))
        else:
            parameters.append(
                ParameterDefinition(
                    f"c{index}",
                    "categorical",
                    categories=("a", "b", "c", "d"),
                )
            )
    return ToyObjective(
        name=f"fixed-{cell.family}-d{cell.dimension}",
        parameters=tuple(parameters),
        optimum=0.0,
        regret_scale=float(cell.dimension),
    )


def make_history(cell: FixedHistoryCell, seed: int) -> FixedHistory:
    """Generate a deterministic history without depending on either implementation's RNG."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    objective = make_objective(cell)
    generator = random.Random(seed)
    rows: list[dict[str, object]] = []
    values: list[float] = []
    for _ in range(cell.observations):
        row = {
            parameter.name: _draw_value(parameter, generator) for parameter in objective.parameters
        }
        rows.append(row)
        values.append(_history_value(objective.parameters, row))
    if len({_row_key(objective.parameters, row) for row in rows}) != len(rows):
        raise RuntimeError("the deterministic fixed-history generator produced duplicate rows")
    payload = {
        "parameters": [parameter.to_upstream_spec() for parameter in objective.parameters],
        "rows": rows,
        "values": values,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return FixedHistory(tuple(rows), tuple(values), hashlib.sha256(encoded).hexdigest())


def run_fixed_history(
    implementation: str,
    cell: FixedHistoryCell,
    settings: RunSettings,
    seed: int,
    *,
    repeats: int = 1,
) -> BenchmarkResult:
    """Time fresh cold suggestions from identical preloaded observations.

    Every repeat reconstructs the optimizer, preloads the same history outside the timed region,
    and performs exactly one model-based suggestion. This avoids accidentally timing LeanHEBO's
    cached second call against upstream HEBO's cold reconstruction.
    """

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if settings.evaluation_budget != cell.observations:
        raise ValueError("settings.evaluation_budget must equal the fixed history size")
    if settings.batch_size != cell.batch_size:
        raise ValueError("settings.batch_size must equal the suggestion batch size")
    if settings.model_lifecycle != "cold":
        raise ValueError("fixed-history matched comparisons require a cold model lifecycle")
    if settings.random_samples > cell.observations:
        raise ValueError("random_samples cannot exceed the fixed history size")

    objective = make_objective(cell)
    history = make_history(cell, seed)
    recorder = PhaseRecorder()
    failures: list[dict[str, object]] = []
    implementation_identity: Mapping[str, object] | None = None
    declared_work: WorkBudget | None = None
    implementation_metrics: list[Mapping[str, object]] = []
    candidate_hashes: list[str] = []
    returned = 0
    invalid = 0
    duplicates = 0
    successful_repeats = 0

    for repeat in range(repeats):
        adapter = make_adapter(implementation, objective, settings, seed)
        if implementation_identity is None:
            implementation_identity = adapter.implementation
            declared_work = adapter.work
        else:
            if dict(adapter.implementation) != dict(implementation_identity):
                raise RuntimeError("implementation identity changed between latency repeats")
            assert declared_work is not None
            assert_matched_work(declared_work, adapter.work)

        stage = "preload"
        try:
            native = _native_history(implementation, objective.parameters, history.rows)
            adapter.observe(Suggested(list(history.rows), native), history.values)
            stage = "suggest"
            with recorder.phase(SUGGEST_PHASE):
                suggested = adapter.suggest(cell.batch_size)
            returned += len(suggested.rows)
            if len(suggested.rows) != cell.batch_size:
                invalid += abs(len(suggested.rows) - cell.batch_size)
                raise RuntimeError(
                    f"optimizer returned {len(suggested.rows)} candidates; "
                    f"expected {cell.batch_size}"
                )
            stage = "candidate-validation"
            _validate_rows(objective.parameters, suggested.rows)
            candidate_keys = [_row_key(objective.parameters, row) for row in suggested.rows]
            seen = {_row_key(objective.parameters, row) for row in history.rows}
            for key in candidate_keys:
                duplicates += int(key in seen)
                seen.add(key)
            stage = "search-work"
            _validate_search_work(suggested.search_report, adapter.work)
            candidate_hashes.append(_candidate_digest(candidate_keys))
            successful_repeats += 1
        except Exception as error:  # Benchmark failures must remain visible in the raw record.
            if stage == "candidate-validation":
                invalid += 1
            failures.append(
                {
                    "repeat": repeat,
                    "stage": stage,
                    "message": str(error),
                    "exception_type": type(error).__name__,
                }
            )
        finally:
            recorder.merge_wall_times(adapter.phase_wall_times())
            implementation_metrics.append(adapter.metrics())

    assert implementation_identity is not None and declared_work is not None
    if len(set(candidate_hashes)) > 1:
        failures.append(
            {
                "stage": "repeat-determinism",
                "message": "identical fixed-history repeats returned different candidates",
                "exception_type": "RuntimeError",
            }
        )
    metrics: dict[str, object] = {
        "history_sha256": history.sha256,
        "history_observations": cell.observations,
        "public_dimension": cell.dimension,
        "family": cell.family,
        "repeats_requested": repeats,
        "repeats_completed": successful_repeats,
        "suggestions_returned": returned,
        "invalid_suggestions": invalid,
        "duplicate_suggestions": duplicates,
        "candidate_sha256": candidate_hashes,
        "implementation_metrics": _group_implementation_metrics(implementation_metrics),
    }
    return BenchmarkResult(
        implementation=implementation_identity,
        suite="fixed-history-latency",
        case=cell.name,
        seed=seed,
        work=declared_work,
        phases=recorder.to_dict(),
        metrics=metrics,
        quality={"normalized_regret": None},
        failures=failures,
    )


def _draw_value(parameter: ParameterDefinition, generator: random.Random) -> object:
    if parameter.kind == "categorical":
        return parameter.categories[generator.randrange(len(parameter.categories))]
    assert parameter.lower is not None and parameter.upper is not None
    if parameter.kind == "integer":
        return generator.randint(int(parameter.lower), int(parameter.upper))
    return (
        float(parameter.lower)
        + (float(parameter.upper) - float(parameter.lower)) * generator.random()
    )


def _history_value(parameters: Sequence[ParameterDefinition], row: Mapping[str, object]) -> float:
    total = 0.0
    for index, parameter in enumerate(parameters):
        value = row[parameter.name]
        if parameter.kind == "categorical":
            categorical_target = parameter.categories[index % len(parameter.categories)]
            total += float(value != categorical_target)
            continue
        assert parameter.lower is not None and parameter.upper is not None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"generated history parameter {parameter.name!r} must be numeric")
        normalized = (float(value) - float(parameter.lower)) / (
            float(parameter.upper) - float(parameter.lower)
        )
        numeric_target = ((index * 37 + 11) % 101) / 100.0
        total += (normalized - numeric_target) ** 2
    return total


def _native_history(
    implementation: str,
    parameters: Sequence[ParameterDefinition],
    rows: Sequence[Mapping[str, object]],
) -> object:
    records = [dict(row) for row in rows]
    if implementation == "leanhebo":
        return records
    if implementation == "upstream-hebo":
        pandas = importlib.import_module("pandas")
        return pandas.DataFrame.from_records(
            records,
            columns=[parameter.name for parameter in parameters],
        )
    raise ValueError(f"unsupported implementation: {implementation}")


def _validate_rows(
    parameters: Sequence[ParameterDefinition], rows: Sequence[Mapping[str, object]]
) -> None:
    expected = {parameter.name for parameter in parameters}
    for row_index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"candidate {row_index} columns differ from the design schema: "
                f"actual={sorted(row)}, expected={sorted(expected)}"
            )
        for parameter in parameters:
            value = row[parameter.name]
            if parameter.kind == "categorical":
                if value not in parameter.categories:
                    raise ValueError(
                        f"candidate {row_index} has an invalid {parameter.name!r} category"
                    )
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"candidate {row_index} parameter {parameter.name!r} must be numeric"
                )
            numeric = float(value)
            assert parameter.lower is not None and parameter.upper is not None
            if not math.isfinite(numeric) or not float(parameter.lower) <= numeric <= float(
                parameter.upper
            ):
                raise ValueError(
                    f"candidate {row_index} parameter {parameter.name!r} is out of bounds"
                )
            if parameter.kind == "integer" and not numeric.is_integer():
                raise ValueError(
                    f"candidate {row_index} parameter {parameter.name!r} is not integral"
                )


def _validate_search_work(report: Mapping[str, int] | None, work: WorkBudget) -> None:
    if report is None:
        raise RuntimeError("fixed-history suggestion did not report actual search work")
    if report.get("candidate_evaluations") != work.search_candidate_evaluations:
        raise RuntimeError(
            "search candidate work differed from its declaration: "
            f"actual={report.get('candidate_evaluations')}, "
            f"declared={work.search_candidate_evaluations}"
        )
    if work.generations is not None and report.get("objective_calls") != work.generations + 1:
        raise RuntimeError(
            "search objective-call work differed from its declaration: "
            f"actual={report.get('objective_calls')}, declared={work.generations + 1}"
        )


def _row_key(parameters: Sequence[ParameterDefinition], row: Mapping[str, object]) -> str:
    return json.dumps(
        [row[parameter.name] for parameter in parameters],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _candidate_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _group_implementation_metrics(
    repeats: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, object] = {"repeats": list(repeats)}
    numerical = _aggregate_numerical_stability(repeats)
    if numerical is not None:
        grouped["numerical_stability"] = numerical
    return grouped


def _aggregate_numerical_stability(
    repeats: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    escalations = 0
    maximum: float | None = None
    give_ups = 0
    random_fallbacks = 0
    reported = False
    for repeat in repeats:
        raw = repeat.get("numerical_stability")
        if raw is None:
            continue
        numerical = _require_mapping(raw, path="implementation_metrics.numerical_stability")
        jitter = _require_mapping(numerical.get("jitter"), path="numerical_stability.jitter")
        fit = _require_mapping(numerical.get("fit"), path="numerical_stability.fit")
        prediction = _require_mapping(
            numerical.get("prediction"), path="numerical_stability.prediction"
        )
        escalations += _non_negative_int(
            jitter.get("escalations"), path="numerical_stability.jitter.escalations"
        )
        give_ups += _non_negative_int(fit.get("give_ups"), path="numerical_stability.fit.give_ups")
        random_fallbacks += _non_negative_int(
            prediction.get("random_fallbacks"),
            path="numerical_stability.prediction.random_fallbacks",
        )
        raw_maximum = jitter.get("maximum_parameter")
        if raw_maximum is not None:
            if isinstance(raw_maximum, bool) or not isinstance(raw_maximum, Real):
                raise TypeError("numerical_stability.jitter.maximum_parameter must be numeric")
            converted = float(raw_maximum)
            if not math.isfinite(converted) or converted < 0:
                raise ValueError(
                    "numerical_stability.jitter.maximum_parameter must be finite and non-negative"
                )
            maximum = converted if maximum is None else max(maximum, converted)
        reported = True
    if not reported:
        return None
    return {
        "jitter": {"escalations": escalations, "maximum_parameter": maximum},
        "fit": {"give_ups": give_ups},
        "prediction": {"random_fallbacks": random_fallbacks},
    }


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    return value


def _non_negative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{path} must be a non-negative integer")
    return value


def _require_axis(name: str, values: Sequence[object], supported: Sequence[object]) -> None:
    if not values:
        raise ValueError(f"{name} must select at least one value")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicate selections")
    invalid = [value for value in values if value not in supported]
    if invalid:
        raise ValueError(f"unsupported {name}: {invalid}; choose from {list(supported)}")
