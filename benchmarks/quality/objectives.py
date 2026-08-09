# SPDX-License-Identifier: MIT

"""Dependency-free analytic objectives shared by LeanHEBO and upstream runs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

ParameterKind = Literal["float", "integer", "categorical"]


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    name: str
    kind: ParameterKind
    lower: float | int | None = None
    upper: float | int | None = None
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter names must be non-empty")
        if self.kind == "categorical":
            if len(self.categories) < 2 or len(set(self.categories)) != len(self.categories):
                raise ValueError("categorical parameters need at least two unique categories")
            if self.lower is not None or self.upper is not None:
                raise ValueError("categorical parameters cannot declare numeric bounds")
        elif self.lower is None or self.upper is None or self.lower >= self.upper:
            raise ValueError("numeric parameters require strictly increasing bounds")

    def to_upstream_spec(self) -> dict[str, object]:
        if self.kind == "categorical":
            return {"name": self.name, "type": "cat", "categories": list(self.categories)}
        return {
            "name": self.name,
            "type": "num" if self.kind == "float" else "int",
            "lb": self.lower,
            "ub": self.upper,
        }


@dataclass(frozen=True, slots=True)
class ToyObjective:
    """A deterministic minimization objective with a predeclared regret scale."""

    name: str
    parameters: tuple[ParameterDefinition, ...]
    optimum: float
    regret_scale: float

    def __post_init__(self) -> None:
        names = [parameter.name for parameter in self.parameters]
        if not self.name or not self.parameters:
            raise ValueError("toy objectives need a name and at least one parameter")
        if len(names) != len(set(names)):
            raise ValueError("toy objective parameter names must be unique")
        if not math.isfinite(self.optimum):
            raise ValueError("the declared optimum must be finite")
        if not math.isfinite(self.regret_scale) or self.regret_scale <= 0:
            raise ValueError("regret_scale must be positive and finite")

    def evaluate(self, rows: Sequence[Mapping[str, object]]) -> list[float]:
        values: list[float] = []
        for row in rows:
            self._validate_row(row)
            value = self._evaluate_one(row)
            if not math.isfinite(value):
                raise ValueError(f"objective {self.name!r} returned a non-finite value")
            values.append(value)
        return values

    def normalized_regret(self, best_value: float) -> float:
        if not math.isfinite(best_value):
            raise ValueError("best_value must be finite")
        return max(0.0, best_value - self.optimum) / self.regret_scale

    def _validate_row(self, row: Mapping[str, object]) -> None:
        missing = [parameter.name for parameter in self.parameters if parameter.name not in row]
        if missing:
            raise ValueError(f"candidate is missing parameters: {', '.join(missing)}")
        for parameter in self.parameters:
            value = row[parameter.name]
            if parameter.kind == "categorical":
                if value not in parameter.categories:
                    raise ValueError(f"invalid category for {parameter.name!r}: {value!r}")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{parameter.name!r} must be numeric")
            numeric = float(value)
            assert parameter.lower is not None and parameter.upper is not None
            if not math.isfinite(numeric) or not parameter.lower <= numeric <= parameter.upper:
                raise ValueError(f"{parameter.name!r} lies outside its declared bounds")
            if parameter.kind == "integer" and numeric != round(numeric):
                raise ValueError(f"{parameter.name!r} is not an integer")

    def _evaluate_one(self, row: Mapping[str, object]) -> float:
        if self.name == "sphere-2d":
            x0 = _numeric_value(row, "x0")
            x1 = _numeric_value(row, "x1")
            return x0 * x0 + x1 * x1
        if self.name == "mixed-3d":
            x = _numeric_value(row, "x")
            depth = round(_numeric_value(row, "depth"))
            category_penalty = {"a": 1.0, "b": 0.0, "c": 2.0}[str(row["kind"])]
            return (x - 1.25) ** 2 + float((depth - 3) ** 2) + category_penalty
        raise RuntimeError(f"objective evaluator is not implemented for {self.name!r}")


def _numeric_value(row: Mapping[str, object], name: str) -> float:
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name!r} must be numeric")
    return float(value)


SPHERE_2D = ToyObjective(
    name="sphere-2d",
    parameters=(
        ParameterDefinition("x0", "float", -5.0, 5.0),
        ParameterDefinition("x1", "float", -5.0, 5.0),
    ),
    optimum=0.0,
    regret_scale=50.0,
)

MIXED_3D = ToyObjective(
    name="mixed-3d",
    parameters=(
        ParameterDefinition("x", "float", -5.0, 5.0),
        ParameterDefinition("depth", "integer", 1, 5),
        ParameterDefinition("kind", "categorical", categories=("a", "b", "c")),
    ),
    optimum=0.0,
    regret_scale=45.0625,
)

OBJECTIVES = {objective.name: objective for objective in (SPHERE_2D, MIXED_3D)}


def get_objective(name: str) -> ToyObjective:
    try:
        return OBJECTIVES[name]
    except KeyError as error:
        choices = ", ".join(sorted(OBJECTIVES))
        raise ValueError(f"unknown toy objective {name!r}; choose one of {choices}") from error
