# SPDX-License-Identifier: MIT

"""Convert supported boundary data to named Python columns."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeAlias, cast

import numpy as np

Columns: TypeAlias = dict[str, list[object]]
_UNSUPPORTED_INPUT = (
    "unsupported input type; expected records, a column mapping, NumPy, Pandas, or Polars"
)


def _is_column_like(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, Sequence) or hasattr(value, "to_list") or hasattr(value, "tolist")


def _column_to_list(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if hasattr(value, "to_list"):
        result = cast(Any, value).to_list()
        return list(result) if isinstance(result, Iterable) else [result]
    elif hasattr(value, "tolist"):
        result = cast(Any, value).tolist()
        return list(result) if isinstance(result, list) else [result]
    elif isinstance(value, Iterable):
        return list(value)
    else:
        raise TypeError("column values must be iterable")


def _validate_lengths(columns: Mapping[str, Sequence[object]]) -> None:
    if len({len(values) for values in columns.values()}) > 1:
        raise ValueError("all input columns must have the same row count")


def _array_columns(value: np.ndarray, names: tuple[str, ...]) -> Columns:
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


def columns_from_input(value: object, names: Sequence[str]) -> Columns:
    expected = tuple(names)
    if isinstance(value, Mapping):
        missing = [name for name in expected if name not in value]
        if missing:
            raise ValueError(f"mapping input is missing columns: {missing}")
        selected = [value[name] for name in expected]
        column_flags = [_is_column_like(item) for item in selected]
        if any(column_flags) and not all(column_flags):
            raise ValueError("mapping input cannot mix scalar values and column sequences")
        columns = (
            {name: _column_to_list(value[name]) for name in expected}
            if all(column_flags)
            else {name: [value[name]] for name in expected}
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return {name: [] for name in expected}
        if not all(isinstance(row, Mapping) for row in value):
            raise TypeError(_UNSUPPORTED_INPUT)
        records = cast(Sequence[Mapping[str, object]], value)
        for row_number, row in enumerate(records):
            missing = [name for name in expected if name not in row]
            if missing:
                raise ValueError(f"record {row_number} is missing columns: {missing}")
        columns = {name: [row[name] for row in records] for name in expected}
    elif isinstance(value, np.ndarray):
        columns = _array_columns(value, expected)
    elif type(value).__module__.split(".", 1)[0] in {"pandas", "polars"}:
        tabular = cast(Any, value)
        missing = [name for name in expected if name not in tabular.columns]
        if missing:
            raise ValueError(f"tabular input is missing columns: {missing}")
        columns = {
            name: (
                tabular.get_column(name).to_list()
                if hasattr(tabular, "get_column")
                else tabular[name].tolist()
            )
            for name in expected
        }
    else:
        raise TypeError(_UNSUPPORTED_INPUT)
    _validate_lengths(columns)
    return columns


__all__ = ["columns_from_input"]
