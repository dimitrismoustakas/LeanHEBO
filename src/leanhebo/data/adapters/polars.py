# SPDX-License-Identifier: MIT

"""Polars input adapter using stable column access APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def polars_to_columns(value: Any, names: Sequence[str]) -> dict[str, list[object]]:
    missing = [name for name in names if name not in value.columns]
    if missing:
        raise ValueError(f"Polars DataFrame is missing columns: {missing}")
    return {name: value.get_column(name).to_list() for name in names}
