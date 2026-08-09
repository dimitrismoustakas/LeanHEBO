# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Public immutable design-space schema."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

import torch

from leanhebo.data.batch import CandidateBatch, EncodedBatch
from leanhebo.space.compiled import CompiledSpace, FixedInput
from leanhebo.space.parameters import Parameter, ParameterLike, parameter_from_spec


class Space:
    """An ordered collection of typed optimization parameters.

    The constructor accepts either positional parameters or one iterable, which
    keeps programmatic schema assembly as convenient as the documented public
    ``Space(Float(...), Integer(...))`` form.
    """

    __slots__ = ("_parameters",)

    def __init__(self, *parameters: object) -> None:
        if (
            len(parameters) == 1
            and not isinstance(parameters[0], Parameter)
            and isinstance(parameters[0], Iterable)
        ):
            normalized = tuple(parameters[0])
        else:
            normalized = parameters
        if not normalized:
            raise ValueError("a design space needs at least one parameter")
        if not all(isinstance(parameter, Parameter) for parameter in normalized):
            raise TypeError("Space accepts Float, Integer, Categorical, and Bool parameters")
        names = tuple(parameter.name for parameter in normalized)
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate parameter names: {duplicates}")
        self._parameters = tuple(cast(ParameterLike, parameter) for parameter in normalized)

    @property
    def parameters(self) -> tuple[ParameterLike, ...]:
        return self._parameters

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self._parameters)

    def __iter__(self) -> Iterator[ParameterLike]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def __repr__(self) -> str:
        arguments = ", ".join(repr(parameter) for parameter in self._parameters)
        return f"Space({arguments})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Space) and self._parameters == other._parameters

    def compile(self, *, dtype: torch.dtype | str = torch.float32) -> CompiledSpace:
        from leanhebo.space.compiled import _dtype_from_value

        return CompiledSpace(self._parameters, dtype=_dtype_from_value(dtype))

    def to_spec(self) -> list[dict[str, object]]:
        """Serialize the schema using JSON/checkpoint-friendly primitives."""

        return [parameter.to_spec() for parameter in self._parameters]

    @classmethod
    def from_spec(cls, specification: Iterable[Mapping[str, Any]]) -> Space:
        parameters = [parameter_from_spec(dict(item)) for item in specification]
        return cls(*parameters)

    # The following conveniences are intentionally thin.  Stateful consumers
    # should compile once and retain CompiledSpace; interactive users can still
    # use a Space directly without learning the internal layer first.
    def sample(
        self,
        count: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        fixed: FixedInput | Mapping[str, object] | None = None,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str | None = None,
    ) -> CandidateBatch:
        return self.compile(dtype=dtype).sample(
            count, seed=seed, generator=generator, fixed=fixed, device=device
        )

    def encode(self, value: object, *, dtype: torch.dtype | str = torch.float32) -> EncodedBatch:
        return self.compile(dtype=dtype).encode(value)
