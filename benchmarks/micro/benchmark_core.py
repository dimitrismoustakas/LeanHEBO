# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from leanhebo.data import ObservationStore
from leanhebo.space import Categorical, Float, Integer, Space
from leanhebo.transforms import OutputTransform

SPACE = Space(
    Float("x", -5.0, 5.0),
    Integer("depth", 1, 20),
    Categorical("kind", ("a", "b", "c")),
).compile()
BATCH = SPACE.sample(128, seed=3)


def test_encode_records(benchmark: object) -> None:
    records = BATCH.to_records()
    benchmark(SPACE.encode, records)  # type: ignore[operator]


def test_direct_candidate_observe(benchmark: object) -> None:
    outcomes = torch.linspace(0, 1, len(BATCH))

    def append() -> None:
        store = ObservationStore(SPACE)
        store.append(BATCH, outcomes)

    benchmark(append)  # type: ignore[operator]


def test_output_transform(benchmark: object) -> None:
    outcomes = torch.linspace(-2, 3, 256).square() - 1
    transform = OutputTransform()
    benchmark(transform.fit_transform, outcomes, force=True)  # type: ignore[operator]
