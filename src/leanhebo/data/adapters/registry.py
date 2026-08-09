# SPDX-License-Identifier: MIT

"""Small explicit registry for supported tabular boundary formats."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np

from leanhebo.data.adapters.numpy import numpy_to_columns
from leanhebo.data.adapters.pandas import pandas_to_columns
from leanhebo.data.adapters.polars import polars_to_columns

Columns: TypeAlias = dict[str, list[object]]
Adapter: TypeAlias = Callable[[Any, Sequence[str]], Columns]
Predicate: TypeAlias = Callable[[object], bool]


def _is_column_like(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, Sequence) or (hasattr(value, "to_list") or hasattr(value, "tolist"))


def _column_to_list(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if hasattr(value, "to_list"):
        result = cast(Any, value).to_list()
        if isinstance(result, Iterable):
            return list(result)
        return [result]
    if hasattr(value, "tolist"):
        result = cast(Any, value).tolist()
        return list(result) if isinstance(result, list) else [result]
    if isinstance(value, Iterable):
        return list(value)
    raise TypeError("column values must be iterable")


@dataclass(slots=True)
class InputAdapterRegistry:
    """Ordered, user-extensible adapter dispatch."""

    _adapters: list[tuple[str, Predicate, Adapter]] = field(default_factory=list)

    def register(
        self,
        name: str,
        predicate: Predicate,
        adapter: Adapter,
        *,
        first: bool = False,
    ) -> None:
        if any(existing_name == name for existing_name, _, _ in self._adapters):
            raise ValueError(f"an input adapter named {name!r} is already registered")
        entry = (name, predicate, adapter)
        if first:
            self._adapters.insert(0, entry)
        else:
            self._adapters.append(entry)

    def columns(self, value: object, names: Sequence[str]) -> Columns:
        expected = tuple(names)
        if isinstance(value, Mapping):
            missing = [name for name in expected if name not in value]
            if missing:
                raise ValueError(f"mapping input is missing columns: {missing}")
            selected = [value[name] for name in expected]
            column_flags = [_is_column_like(item) for item in selected]
            if any(column_flags) and not all(column_flags):
                raise ValueError("mapping input cannot mix scalar values and column sequences")
            if all(column_flags):
                columns = {name: _column_to_list(value[name]) for name in expected}
            else:
                columns = {name: [value[name]] for name in expected}
            self._validate_lengths(columns)
            return columns
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                return {name: [] for name in expected}
            if all(isinstance(row, Mapping) for row in value):
                records = cast(Sequence[Mapping[str, object]], value)
                for row_number, row in enumerate(records):
                    missing = [name for name in expected if name not in row]
                    if missing:
                        raise ValueError(f"record {row_number} is missing columns: {missing}")
                return {name: [row[name] for row in records] for name in expected}
        for _, predicate, adapter in self._adapters:
            if predicate(value):
                columns = adapter(value, expected)
                self._validate_lengths(columns)
                return columns
        raise TypeError(
            "unsupported input type; expected records, a column mapping, NumPy, Pandas, or Polars"
        )

    @staticmethod
    def _validate_lengths(columns: Mapping[str, Sequence[object]]) -> None:
        lengths = {len(values) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError("all input columns must have the same row count")


def _module_starts_with(value: object, module: str) -> bool:
    return type(value).__module__.split(".", 1)[0] == module


DEFAULT_ADAPTERS = InputAdapterRegistry()
DEFAULT_ADAPTERS.register("numpy", lambda value: isinstance(value, np.ndarray), numpy_to_columns)
DEFAULT_ADAPTERS.register(
    "pandas", lambda value: _module_starts_with(value, "pandas"), pandas_to_columns
)
DEFAULT_ADAPTERS.register(
    "polars", lambda value: _module_starts_with(value, "polars"), polars_to_columns
)


def columns_from_input(value: object, names: Sequence[str]) -> Columns:
    return DEFAULT_ADAPTERS.columns(value, names)
