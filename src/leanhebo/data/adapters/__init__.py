# SPDX-License-Identifier: MIT

"""Convert supported boundary data to named Python columns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import numpy as np
import torch

Columns: TypeAlias = dict[str, list[object]]
Records: TypeAlias = list[dict[str, object]]
_UNSUPPORTED_INPUT = "unsupported input type; expected records, a column mapping, or NumPy"


def _is_column_like(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.ndim > 0
    return isinstance(value, Sequence)


def _column_to_list(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise TypeError("column values must be sequences or NumPy arrays")


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
    else:
        raise TypeError(_UNSUPPORTED_INPUT)
    _validate_lengths(columns)
    return columns


def records_from_input(value: object, names: Sequence[str]) -> Records:
    """Convert boundary data to records while permitting absent mapping fields."""

    expected = tuple(names)
    expected_set = set(expected)
    if isinstance(value, Mapping):
        unknown = sorted(str(name) for name in set(value).difference(expected_set))
        if unknown:
            raise ValueError(f"mapping input contains unknown columns: {unknown}")
        present = tuple(name for name in expected if name in value)
        selected = [value[name] for name in present]
        column_flags = [_is_column_like(item) for item in selected]
        if any(column_flags) and not all(column_flags):
            raise ValueError("mapping input cannot mix scalar values and column sequences")
        if selected and all(column_flags):
            columns = {name: _column_to_list(value[name]) for name in present}
            _validate_lengths(columns)
            row_count = len(next(iter(columns.values())))
            return [{name: columns[name][row] for name in present} for row in range(row_count)]
        return [{name: value[name] for name in present}]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return []
        if not all(isinstance(row, Mapping) for row in value):
            raise TypeError(_UNSUPPORTED_INPUT)
        for row_number, row in enumerate(cast(Sequence[Mapping[str, object]], value)):
            unknown = sorted(str(name) for name in set(row).difference(expected_set))
            if unknown:
                raise ValueError(f"record {row_number} contains unknown columns: {unknown}")
        return [
            {name: row[name] for name in expected if name in row}
            for row in cast(Sequence[Mapping[str, object]], value)
        ]
    if isinstance(value, np.ndarray) and value.dtype.names is not None:
        unknown = sorted(set(value.dtype.names).difference(expected_set))
        if unknown:
            raise ValueError(f"NumPy structured array contains unknown columns: {unknown}")
    columns = columns_from_input(value, expected)
    if not columns:
        return []
    row_count = len(next(iter(columns.values())))
    return [{name: columns[name][row] for name in expected} for row in range(row_count)]


__all__ = ["columns_from_input", "records_from_input"]
