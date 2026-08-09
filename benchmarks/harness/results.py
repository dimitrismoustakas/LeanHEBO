# SPDX-License-Identifier: MIT

"""Versioned benchmark result records and low-overhead phase timing."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TypeAlias

from benchmarks.harness.work import Scalar, WorkBudget

JsonValue: TypeAlias = Scalar | list["JsonValue"] | dict[str, "JsonValue"]
PhaseData: TypeAlias = dict[str, dict[str, list[float]]]


def _json_value(value: object, *, path: str = "value") -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}")


@dataclass(slots=True)
class PhaseRecorder:
    """Collect repeatable wall and process-CPU samples for named phases."""

    _wall_seconds: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _process_cpu_seconds: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not name or name.strip() != name:
            raise ValueError("phase names must be non-empty and have no surrounding whitespace")
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            yield
        finally:
            self._process_cpu_seconds[name].append(time.process_time() - cpu_start)
            self._wall_seconds[name].append(time.perf_counter() - wall_start)

    def merge_wall_times(self, phases: Mapping[str, Sequence[float]]) -> None:
        """Merge implementation-provided wall samples, validating them before recording."""

        for name, samples in phases.items():
            if not isinstance(name, str) or not name:
                raise ValueError("phase names must be non-empty strings")
            converted = [float(sample) for sample in samples]
            if any(not math.isfinite(sample) or sample < 0 for sample in converted):
                raise ValueError(f"phase {name!r} contains an invalid duration")
            self._wall_seconds[name].extend(converted)

    def to_dict(self) -> PhaseData:
        names = sorted(self._wall_seconds.keys() | self._process_cpu_seconds.keys())
        return {
            name: {
                "wall_seconds": list(self._wall_seconds[name]),
                "process_cpu_seconds": list(self._process_cpu_seconds[name]),
            }
            for name in names
        }


def collect_runtime_metadata() -> dict[str, JsonValue]:
    """Collect enough process and Torch metadata to make a timing record auditable."""

    metadata: dict[str, JsonValue] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "executable": sys.executable,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "packages": {
            distribution: _distribution_version(distribution)
            for distribution in (
                "torch",
                "gpytorch",
                "numpy",
                "pandas",
                "pymoo",
                "GPy",
                "HEBO",
                "leanhebo",
            )
        },
    }
    try:
        import torch
    except ImportError:  # pragma: no cover - only relevant in the isolated upstream environment
        metadata["torch"] = None
        return metadata

    torch_metadata: dict[str, JsonValue] = {
        "version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "build_config": torch.__config__.show(),
    }
    if torch.cuda.is_available():
        torch_metadata["cuda_device_count"] = torch.cuda.device_count()
        torch_metadata["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    metadata["torch"] = torch_metadata
    return metadata


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One validated raw timing or quality result."""

    implementation: Mapping[str, object]
    suite: str
    case: str
    seed: int
    work: WorkBudget
    phases: Mapping[str, Mapping[str, Sequence[float]]]
    metrics: Mapping[str, object] = field(default_factory=dict)
    quality: Mapping[str, object] | None = None
    failures: Sequence[Mapping[str, object]] = ()
    runtime: Mapping[str, object] = field(default_factory=collect_runtime_metadata)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at_utc: str = field(
        # datetime.UTC is unavailable in the pinned baseline's Python 3.10 runtime.
        default_factory=lambda: datetime.now(timezone.utc).isoformat()  # noqa: UP017
    )

    def __post_init__(self) -> None:
        for name, value in (("suite", self.suite), ("case", self.case), ("run_id", self.run_id)):
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, object] = {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "implementation": self.implementation,
            "benchmark": {"suite": self.suite, "case": self.case, "seed": self.seed},
            "work": self.work.to_dict(),
            "runtime": self.runtime,
            "phases": self.phases,
            "metrics": self.metrics,
            "quality": self.quality,
            "failures": self.failures,
        }
        converted = _json_value(result, path="result")
        assert isinstance(converted, dict)
        return converted


def write_result(result: BenchmarkResult, destination: str | Path) -> Path:
    """Atomically write a result without overwriting an existing raw record."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark result: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
