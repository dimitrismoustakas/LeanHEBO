# SPDX-License-Identifier: MIT

"""Low-overhead diagnostics and named phase timing."""

from __future__ import annotations

import math
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from leanhebo.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class FitReport:
    """Outcome of one exact-GP fit or update."""

    kind: str
    observations: int
    requested_steps: int
    completed_steps: int
    final_loss: float | None
    wall_time: float
    maximum_jitter: float
    jitter_retries: int
    early_stopped: bool = False
    failure: str | None = None


@dataclass(slots=True)
class Diagnostics:
    """Accumulates phase durations, counters, and recent model reports."""

    runtime: RuntimeConfig
    phase_seconds: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    counters: Counter[str] = field(default_factory=Counter)
    fit_reports: list[FitReport] = field(default_factory=list)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        synchronize = self.runtime.synchronize_device_for_timing and self.runtime.device.startswith(
            "cuda"
        )
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(self.runtime.device))
        start = time.perf_counter()
        try:
            yield
        finally:
            if synchronize and torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(self.runtime.device))
            self.phase_seconds[name].append(time.perf_counter() - start)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def add_fit_report(self, report: FitReport) -> None:
        self.fit_reports.append(report)

    def effective_runtime(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "device": self.runtime.device,
            "dtype": self.runtime.dtype,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
        }
        if self.runtime.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(self.runtime.device)
            result["cuda_device_name"] = torch.cuda.get_device_name(device)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_seconds": dict(self.phase_seconds),
            "counters": dict(self.counters),
            "fit_reports": [asdict(report) for report in self.fit_reports],
            "effective_runtime": self.effective_runtime(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint-safe accumulated diagnostics without live runtime metadata."""

        return {
            "schema_version": 1,
            "phase_seconds": {name: list(samples) for name, samples in self.phase_seconds.items()},
            "counters": dict(self.counters),
            "fit_reports": [asdict(report) for report in self.fit_reports],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore validated counters, timings, and fit reports."""

        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("unsupported diagnostics state schema")
        phases = state.get("phase_seconds")
        counters = state.get("counters")
        reports = state.get("fit_reports")
        if (
            not isinstance(phases, Mapping)
            or not isinstance(counters, Mapping)
            or not isinstance(reports, list)
        ):
            raise TypeError("diagnostics state is malformed")
        restored_phases: defaultdict[str, list[float]] = defaultdict(list)
        for name, samples in phases.items():
            if not isinstance(name, str) or not isinstance(samples, list):
                raise TypeError("diagnostic phase state is malformed")
            values = [float(sample) for sample in samples]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("diagnostic phase durations must be finite and non-negative")
            restored_phases[name] = values
        restored_counters: Counter[str] = Counter()
        for name, value in counters.items():
            if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("diagnostic counters must map strings to integers")
            restored_counters[name] = value
        restored_reports = []
        for report in reports:
            if not isinstance(report, Mapping):
                raise TypeError("diagnostic fit report state is malformed")
            restored_reports.append(FitReport(**dict(report)))
        self.phase_seconds = restored_phases
        self.counters = restored_counters
        self.fit_reports = restored_reports
