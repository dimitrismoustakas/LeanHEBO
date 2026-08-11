# SPDX-License-Identifier: MIT

"""Design-space public API."""

from leanhebo.space.compiled import CompiledSpace, FixedInput
from leanhebo.space.parameters import Bool, Categorical, Float, Integer, Parameter
from leanhebo.space.space import Space

__all__ = [
    "Bool",
    "Categorical",
    "CompiledSpace",
    "FixedInput",
    "Float",
    "Integer",
    "Parameter",
    "Space",
]
