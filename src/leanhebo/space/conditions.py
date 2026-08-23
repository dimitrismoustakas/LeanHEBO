# SPDX-License-Identifier: MIT

"""Immutable, serializable expressions for conditional parameter activity."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any as TypingAny
from typing import ClassVar, TypeAlias

ConditionValue: TypeAlias = str | int | float | bool | None


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("condition parameter names must be non-empty strings")
    return name


def _normalize_value(value: object) -> ConditionValue:
    if type(value).__module__.split(".", 1)[0] == "numpy" and hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("condition values must be scalar JSON-safe primitives")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("floating condition values must be finite")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_numeric(value: object) -> float | int:
    value = _normalize_value(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("ordered condition thresholds must be real numbers")
    return value


def _value_identity(value: ConditionValue) -> tuple[type[object], ConditionValue]:
    return type(value), value


def _spec_key(condition: Condition) -> str:
    return json.dumps(
        condition.to_spec(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class Condition(ABC):
    """Base class for one immutable activity expression."""

    @property
    @abstractmethod
    def references(self) -> frozenset[str]:
        """Parameter names read by this expression."""

    @abstractmethod
    def to_spec(self) -> dict[str, object]:
        """Return a JSON-safe expression specification."""


@dataclass(frozen=True, slots=True, eq=False)
class _ValueAtom(Condition):
    name: str
    value: ConditionValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "value", _normalize_value(self.value))

    @property
    def references(self) -> frozenset[str]:
        return frozenset((self.name,))

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and isinstance(other, _ValueAtom)
            and self.name == other.name
            and _value_identity(self.value) == _value_identity(other.value)
        )

    def __hash__(self) -> int:
        return hash((type(self), self.name, _value_identity(self.value)))


@dataclass(frozen=True, slots=True, eq=False)
class Eq(_ValueAtom):
    """True when an active discrete parameter equals ``value``."""

    def to_spec(self) -> dict[str, object]:
        return {"type": "eq", "name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True, eq=False)
class NotEqual(_ValueAtom):
    """True when an active discrete parameter does not equal ``value``."""

    def to_spec(self) -> dict[str, object]:
        return {"type": "not_equal", "name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True, init=False, eq=False)
class In(Condition):
    """True when an active discrete parameter belongs to ``values``."""

    name: str
    values: tuple[ConditionValue, ...]

    def __init__(self, name: str, values: Iterable[object]) -> None:
        normalized: list[ConditionValue] = []
        for value in values:
            item = _normalize_value(value)
            if any(_value_identity(item) == _value_identity(previous) for previous in normalized):
                continue
            normalized.append(item)
        if not normalized:
            raise ValueError("In requires at least one value")
        normalized.sort(
            key=lambda item: json.dumps(
                item,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        object.__setattr__(self, "name", _validate_name(name))
        object.__setattr__(self, "values", tuple(normalized))

    @property
    def references(self) -> frozenset[str]:
        return frozenset((self.name,))

    def to_spec(self) -> dict[str, object]:
        return {"type": "in", "name": self.name, "values": list(self.values)}

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, In)
            and self.name == other.name
            and tuple(_value_identity(value) for value in self.values)
            == tuple(_value_identity(value) for value in other.values)
        )

    def __hash__(self) -> int:
        identities = tuple(_value_identity(value) for value in self.values)
        return hash((type(self), self.name, identities))


@dataclass(frozen=True, slots=True)
class _OrderedAtom(Condition):
    name: str
    value: float | int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "value", _normalize_numeric(self.value))

    @property
    def references(self) -> frozenset[str]:
        return frozenset((self.name,))


@dataclass(frozen=True, slots=True)
class LessThan(_OrderedAtom):
    def to_spec(self) -> dict[str, object]:
        return {"type": "less_than", "name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class LessEqual(_OrderedAtom):
    def to_spec(self) -> dict[str, object]:
        return {"type": "less_equal", "name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class GreaterThan(_OrderedAtom):
    def to_spec(self) -> dict[str, object]:
        return {"type": "greater_than", "name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class GreaterEqual(_OrderedAtom):
    def to_spec(self) -> dict[str, object]:
        return {"type": "greater_equal", "name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True, init=False)
class _Combination(Condition):
    conditions: tuple[Condition, ...]

    operation: ClassVar[str] = ""

    def __init__(self, *conditions: Condition) -> None:
        flattened: list[Condition] = []
        for condition in conditions:
            if not isinstance(condition, Condition):
                raise TypeError("All and Any require Condition operands")
            if isinstance(condition, _Combination) and type(condition) is type(self):
                flattened.extend(condition.conditions)
            else:
                flattened.append(condition)
        if not flattened:
            raise ValueError("All and Any require at least one condition")
        unique = {condition: condition for condition in flattened}
        ordered = tuple(sorted(unique.values(), key=_spec_key))
        object.__setattr__(self, "conditions", ordered)

    @property
    def references(self) -> frozenset[str]:
        return frozenset().union(*(condition.references for condition in self.conditions))

    def to_spec(self) -> dict[str, object]:
        return {
            "type": self.operation,
            "conditions": [condition.to_spec() for condition in self.conditions],
        }


@dataclass(frozen=True, slots=True, init=False)
class All(_Combination):
    """Conjunction of one or more guarded conditions."""

    operation = "all"


@dataclass(frozen=True, slots=True, init=False)
class Any(_Combination):
    """Disjunction of one or more guarded conditions."""

    operation = "any"


def normalize_condition(condition: Condition) -> Condition:
    """Return the canonical syntactic form used for compilation and grouping."""

    if isinstance(condition, In) and len(condition.values) == 1:
        return Eq(condition.name, condition.values[0])
    if isinstance(condition, All):
        normalized_all = All(*(normalize_condition(child) for child in condition.conditions))
        return (
            normalized_all.conditions[0] if len(normalized_all.conditions) == 1 else normalized_all
        )
    if isinstance(condition, Any):
        normalized_any = Any(*(normalize_condition(child) for child in condition.conditions))
        return (
            normalized_any.conditions[0] if len(normalized_any.conditions) == 1 else normalized_any
        )
    return condition


def condition_from_spec(spec: Mapping[str, TypingAny]) -> Condition:
    """Reconstruct one condition from its JSON-safe specification."""

    item = dict(spec)
    try:
        type_name = item.pop("type")
    except KeyError as error:
        raise ValueError("missing condition-spec field: type") from error
    constructors: dict[str, type[_ValueAtom] | type[_OrderedAtom]] = {
        "eq": Eq,
        "not_equal": NotEqual,
        "less_than": LessThan,
        "less_equal": LessEqual,
        "greater_than": GreaterThan,
        "greater_equal": GreaterEqual,
    }
    if type_name in constructors:
        try:
            name = item.pop("name")
            value = item.pop("value")
        except KeyError as error:
            raise ValueError(f"missing condition-spec field: {error.args[0]}") from error
        if item:
            raise ValueError(f"unexpected condition fields: {sorted(item)}")
        return constructors[str(type_name)](name, value)
    if type_name == "in":
        try:
            name = item.pop("name")
            values = item.pop("values")
        except KeyError as error:
            raise ValueError(f"missing condition-spec field: {error.args[0]}") from error
        if item:
            raise ValueError(f"unexpected condition fields: {sorted(item)}")
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes, bytearray)):
            raise TypeError("condition 'values' must be an iterable")
        return In(name, values)
    if type_name in {"all", "any"}:
        try:
            children = item.pop("conditions")
        except KeyError as error:
            raise ValueError("missing condition-spec field: conditions") from error
        if item:
            raise ValueError(f"unexpected condition fields: {sorted(item)}")
        if not isinstance(children, Iterable) or isinstance(children, (str, bytes, bytearray)):
            raise TypeError("condition 'conditions' must be an iterable")
        parsed: list[Condition] = []
        for child in children:
            if not isinstance(child, Mapping):
                raise TypeError("nested condition specs must be mappings")
            parsed.append(condition_from_spec(child))
        return All(*parsed) if type_name == "all" else Any(*parsed)
    raise ValueError(f"unsupported condition type {type_name!r}")


__all__ = [
    "All",
    "Any",
    "Condition",
    "Eq",
    "GreaterEqual",
    "GreaterThan",
    "In",
    "LessEqual",
    "LessThan",
    "NotEqual",
    "condition_from_spec",
    "normalize_condition",
]
