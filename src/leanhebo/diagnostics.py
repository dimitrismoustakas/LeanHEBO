# SPDX-License-Identifier: MIT

"""Low-overhead diagnostics and named phase timing."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

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
