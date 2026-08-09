# SPDX-License-Identifier: MIT

"""A small mixed-variable LeanHEBO loop."""

from __future__ import annotations

import numpy as np

from leanhebo import LeanHEBO
from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig
from leanhebo.space import Bool, Categorical, Float, Integer, Space


def objective(row: dict[str, object]) -> float:
    x = float(row["x"])
    depth = int(row["depth"])
    category_penalty = 0.0 if row["activation"] == "gelu" else 0.25
    bias_penalty = 0.0 if bool(row["use_bias"]) else 0.1
    return (x - 0.3) ** 2 + (depth - 6) ** 2 / 25 + category_penalty + bias_penalty


space = Space(
    Float("x", -2.0, 2.0),
    Integer("depth", 1, 12),
    Categorical("activation", ("relu", "gelu", "silu")),
    Bool("use_bias"),
)
config = LeanHEBOConfig(
    runtime=RuntimeConfig(seed=7),
    gp=GPConfig(initial_steps=30, update_steps=5),
    search=SearchConfig(population_size=64, generations=30),
)
optimizer = LeanHEBO(space, config=config)

for _ in range(12):
    batch = optimizer.suggest(3)
    y = np.asarray([objective(record) for record in batch.to_records()])
    optimizer.observe(batch, y)

print(optimizer.best_x.to_records()[0], optimizer.best_y)
