# SPDX-License-Identifier: MIT

"""Optional table adapters remain outside the optimization hot path."""

from __future__ import annotations

import numpy as np

from leanhebo import LeanHEBO
from leanhebo.config import LeanHEBOConfig, RuntimeConfig
from leanhebo.space import Categorical, Float, Space

space = Space(Float("x", -1.0, 1.0), Categorical("kind", ("left", "right")))
optimizer = LeanHEBO(space, config=LeanHEBOConfig(runtime=RuntimeConfig(seed=3)))

candidates = optimizer.suggest(4)
pandas_frame = candidates.to_pandas()
polars_frame = candidates.to_polars()
assert pandas_frame.shape == polars_frame.shape

values = np.square(pandas_frame["x"].to_numpy())
optimizer.observe(candidates, values)  # reuses CandidateBatch tensors
