# SPDX-License-Identifier: MIT

"""Pandas input adapter with no import-time Pandas dependency."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def pandas_to_columns(value: Any, names: Sequence[str]) -> dict[str, list[object]]:
    missing = [name for name in names if name not in value.columns]
    if missing:
        raise ValueError(f"Pandas DataFrame is missing columns: {missing}")
    return {name: value[name].tolist() for name in names}
