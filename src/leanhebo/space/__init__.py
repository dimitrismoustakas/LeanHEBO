# SPDX-License-Identifier: MIT

"""Design-space public API."""

from leanhebo.space.compiled import CompiledSpace, FixedInput
from leanhebo.space.conditions import (
    All,
    Any,
    Condition,
    Eq,
    GreaterEqual,
    GreaterThan,
    In,
    LessEqual,
    LessThan,
    NotEqual,
)
from leanhebo.space.parameters import Bool, Categorical, Float, Integer, Parameter
from leanhebo.space.space import Space

__all__ = [
    "All",
    "Any",
    "Bool",
    "Categorical",
    "CompiledSpace",
    "Condition",
    "Eq",
    "FixedInput",
    "Float",
    "GreaterEqual",
    "GreaterThan",
    "In",
    "Integer",
    "LessEqual",
    "LessThan",
    "NotEqual",
    "Parameter",
    "Space",
]
