# SPDX-License-Identifier: MIT

"""NumPy input adapter."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def numpy_to_columns(value: np.ndarray, names: Sequence[str]) -> dict[str, list[object]]:
    """Convert a regular or structured ndarray to named Python columns."""

    if value.dtype.names is not None:
        missing = [name for name in names if name not in value.dtype.names]
        if missing:
            raise ValueError(f"NumPy structured array is missing columns: {missing}")
        return {name: value[name].reshape(-1).tolist() for name in names}
    if value.ndim == 1:
        if value.size != len(names):
            raise ValueError(
                f"one-dimensional NumPy input must contain {len(names)} values, got {value.size}"
            )
        value = value.reshape(1, -1)
    if value.ndim != 2:
        raise ValueError("NumPy input must have one or two dimensions")
    if value.shape[1] != len(names):
        raise ValueError(f"NumPy input must have {len(names)} columns, got {value.shape[1]}")
    return {name: value[:, index].tolist() for index, name in enumerate(names)}
