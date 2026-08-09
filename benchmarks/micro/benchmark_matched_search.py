# SPDX-License-Identifier: MIT

"""Structural search benchmark with an asserted, fixed candidate-work budget."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.harness.work import WorkBudget
from leanhebo.search import TorchNSGA2

POPULATION = 32
GENERATIONS = 4
DIMENSIONS = 5
WORK = WorkBudget(
    objective_evaluations=0,
    batch_size=1,
    population_size=POPULATION,
    generations=GENERATIONS,
)


@dataclass(slots=True)
class _ObjectiveCounter:
    calls: int = 0
    candidates: int = 0

    def __call__(self, population: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.candidates += population.shape[0]
        shifted = population - 0.25
        return torch.stack(
            (
                population.square().sum(dim=1),
                shifted.square().sum(dim=1),
                population.abs().sum(dim=1),
            ),
            dim=1,
        )


def _run_search() -> tuple[int, int]:
    counter = _ObjectiveCounter()
    optimizer = TorchNSGA2(
        population_size=POPULATION,
        generations=GENERATIONS,
        eliminate_duplicate_points=False,
    )
    result = optimizer.minimize(
        counter,
        torch.full((DIMENSIONS,), -2.0),
        torch.full((DIMENSIONS,), 2.0),
        generator=torch.Generator().manual_seed(17),
    )
    assert result.generations == GENERATIONS
    return counter.calls, counter.candidates


def test_torch_nsga2_fixed_candidate_work(benchmark: object) -> None:
    calls, candidates = benchmark(_run_search)  # type: ignore[operator]
    assert calls == GENERATIONS + 1
    assert candidates == WORK.expected_search_evaluations
