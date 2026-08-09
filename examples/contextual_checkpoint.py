# SPDX-License-Identifier: MIT

"""Contextual suggestions and deterministic checkpoint continuation."""

from __future__ import annotations

from leanhebo import LeanHEBO
from leanhebo.config import LeanHEBOConfig, RuntimeConfig
from leanhebo.space import Categorical, Float, Space

space = Space(Float("dose", 0.0, 1.0), Categorical("site", ("north", "south")))
optimizer = LeanHEBO(space, config=LeanHEBOConfig(runtime=RuntimeConfig(seed=19)))

south_batch = optimizer.suggest(4, fix_input={"site": "south"})
assert all(record["site"] == "south" for record in south_batch.to_records())

optimizer.save("contextual.leanhebo")
restored = LeanHEBO.load("contextual.leanhebo", map_location="cpu")
next_batch = restored.suggest(2, fix_input={"site": "south"})
print(next_batch.to_records())
