# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Typed parameter definitions for LeanHEBO design spaces.

Parameters describe values in the user domain and their corresponding numeric
optimization coordinates.  The classes are deliberately immutable: compilation
can safely cache all derived metadata without having to account for later schema
mutation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from numbers import Real
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias, cast

import torch

from leanhebo.space.conditions import Condition, condition_from_spec, normalize_condition

Primitive: TypeAlias = str | int | float | bool | None


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("parameter names must be non-empty strings")


def _as_finite_float(value: object, *, parameter: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{parameter!r} expects a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{parameter!r} expects a finite value")
    return result


def _as_integer(value: object, *, parameter: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{parameter!r} expects an integer, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{parameter!r} expects an integer, got {value!r}")
    return int(numeric)


@dataclass(frozen=True, slots=True, repr=False)
class Parameter(ABC):
    """Base class shared by all public parameter types."""

    name: str
    active_when: Condition | None = field(default=None, kw_only=True, repr=False)
    type_name: ClassVar[str]

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if self.active_when is not None and not isinstance(self.active_when, Condition):
            raise TypeError("active_when must be a Condition or None")
        if self.active_when is not None:
            object.__setattr__(self, "active_when", normalize_condition(self.active_when))

    @property
    @abstractmethod
    def is_categorical(self) -> bool:
        """Whether this parameter is stored in the categorical tensor."""

    @property
    @abstractmethod
    def optimization_bounds(self) -> tuple[float, float]:
        """Bounds in dense optimization coordinates."""

    @property
    def is_integer(self) -> bool:
        return False

    @property
    def is_discrete_after_transform(self) -> bool:
        return self.is_categorical

    @abstractmethod
    def encode_values(self, values: list[object], *, dtype: torch.dtype) -> torch.Tensor:
        """Encode validated user values to a one-dimensional tensor."""

    @abstractmethod
    def decode_values(self, values: torch.Tensor) -> tuple[object, ...]:
        """Decode a one-dimensional optimization-coordinate tensor."""

    @abstractmethod
    def to_spec(self) -> dict[str, object]:
        """Return a checkpoint-friendly schema description."""

    def _with_condition(self, specification: dict[str, object]) -> dict[str, object]:
        if self.active_when is not None:
            specification["active_when"] = self.active_when.to_spec()
        return specification

    def __repr__(self) -> str:
        arguments = [
            f"{item.name}={getattr(self, item.name)!r}" for item in fields(self) if item.repr
        ]
        if self.active_when is not None:
            arguments.append(f"active_when={self.active_when!r}")
        return f"{type(self).__name__}({', '.join(arguments)})"


@dataclass(frozen=True, slots=True, repr=False)
class Float(Parameter):
    """A bounded real parameter, optionally optimized on a logarithmic scale."""

    low: float
    high: float
    log: bool = False
    base: float = 10.0
    type_name: ClassVar[str] = "float"

    def __post_init__(self) -> None:
        Parameter.__post_init__(self)
        low = _as_finite_float(self.low, parameter=self.name)
        high = _as_finite_float(self.high, parameter=self.name)
        base = _as_finite_float(self.base, parameter=f"{self.name}.base")
        if not low < high:
            raise ValueError(f"{self.name!r} requires low < high")
        if self.log and low <= 0:
            raise ValueError(f"log-scaled parameter {self.name!r} requires low > 0")
        if self.log and base <= 1:
            raise ValueError("a logarithm base must be greater than one")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "base", base)

    @property
    def is_categorical(self) -> bool:
        return False

    @property
    def optimization_bounds(self) -> tuple[float, float]:
        if not self.log:
            return self.low, self.high
        denominator = math.log(self.base)
        return math.log(self.low) / denominator, math.log(self.high) / denominator

    def encode_values(self, values: list[object], *, dtype: torch.dtype) -> torch.Tensor:
        validated = [_as_finite_float(value, parameter=self.name) for value in values]
        for value in validated:
            if value < self.low or value > self.high:
                raise ValueError(
                    f"value {value!r} for {self.name!r} lies outside [{self.low}, {self.high}]"
                )
        result = torch.tensor(validated, dtype=dtype)
        if self.log:
            result = torch.log(result) / math.log(self.base)
            lower, upper = self.optimization_bounds
            result = result.clamp(lower, upper)
        return result

    def decode_values(self, values: torch.Tensor) -> tuple[object, ...]:
        decoded = values.detach().to(device="cpu", dtype=torch.float64)
        if self.log:
            decoded = torch.pow(torch.tensor(self.base, dtype=torch.float64), decoded)
        decoded = decoded.clamp(self.low, self.high)
        return tuple(decoded.tolist())

    def to_spec(self) -> dict[str, object]:
        return self._with_condition(
            {
                "name": self.name,
                "type": self.type_name,
                "low": self.low,
                "high": self.high,
                "log": self.log,
                "base": self.base,
            }
        )


@dataclass(frozen=True, slots=True, repr=False)
class Integer(Parameter):
    """A bounded integer parameter with HEBO-compatible transform variants.

    ``step`` represents domains such as ``4, 8, 12, 16`` using an integer
    index internally.  ``log=True`` implements HEBO's ``pow_int`` behavior:
    arbitrary integers are optimized in logarithmic coordinates.  Setting
    ``exponent=True`` restricts the domain to integral powers of ``base`` and
    implements HEBO's ``int_exponent`` behavior.
    """

    low: int
    high: int
    step: int = 1
    log: bool = False
    base: float = 10.0
    exponent: bool = False
    type_name: ClassVar[str] = "integer"

    def __post_init__(self) -> None:
        Parameter.__post_init__(self)
        low = _as_integer(self.low, parameter=self.name)
        high = _as_integer(self.high, parameter=self.name)
        step = _as_integer(self.step, parameter=f"{self.name}.step")
        base = _as_finite_float(self.base, parameter=f"{self.name}.base")
        log = self.log
        if low > high:
            raise ValueError(f"{self.name!r} requires low <= high")
        if step <= 0:
            raise ValueError("integer step must be positive")
        if (log or self.exponent) and step != 1:
            raise ValueError("step cannot be combined with log or exponent transforms")
        if (log or self.exponent) and low < 1:
            raise ValueError("log-scaled integer parameters require low >= 1")
        if (log or self.exponent) and base <= 1:
            raise ValueError("a logarithm base must be greater than one")
        if not log and not self.exponent and (high - low) % step != 0:
            raise ValueError("high - low must be divisible by step")
        if self.exponent:
            if not base.is_integer() or base < 2:
                raise ValueError("exponent integer parameters require an integral base >= 2")
            low_exp = math.log(low, base)
            high_exp = math.log(high, base)
            if not math.isclose(low_exp, round(low_exp), abs_tol=1e-12) or not math.isclose(
                high_exp, round(high_exp), abs_tol=1e-12
            ):
                raise ValueError("exponent integer bounds must be exact integral powers of base")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "log", log)

    @property
    def is_categorical(self) -> bool:
        return False

    @property
    def is_integer(self) -> bool:
        return True

    @property
    def is_discrete_after_transform(self) -> bool:
        return not self.log or self.exponent

    @property
    def mode(self) -> str:
        if self.exponent:
            return "exponent"
        if self.log:
            return "power"
        if self.step != 1:
            return "step"
        return "linear"

    @property
    def optimization_bounds(self) -> tuple[float, float]:
        if self.exponent:
            return (
                float(round(math.log(self.low, self.base))),
                float(round(math.log(self.high, self.base))),
            )
        if self.log:
            return (0.0, 0.0) if self.low == self.high else (0.0, 1.0)
        if self.step != 1:
            return 0.0, float((self.high - self.low) // self.step)
        return float(self.low), float(self.high)

    def _validate_user_integer(self, value: object) -> int:
        integer = _as_integer(value, parameter=self.name)
        if integer < self.low or integer > self.high:
            raise ValueError(
                f"value {integer!r} for {self.name!r} lies outside [{self.low}, {self.high}]"
            )
        if self.step != 1 and (integer - self.low) % self.step:
            raise ValueError(
                f"value {integer!r} for {self.name!r} does not align to step {self.step}"
            )
        if self.exponent:
            exponent = math.log(integer, self.base)
            if not math.isclose(exponent, round(exponent), abs_tol=1e-12):
                raise ValueError(
                    f"value {integer!r} for {self.name!r} is not an integral power of {self.base:g}"
                )
        return integer

    def _encode_log_values(self, values: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if self.low == self.high:
            return torch.zeros(values.shape, dtype=dtype, device=values.device)
        semantic = values.to(dtype=torch.float64)
        log_low = math.log(self.low)
        log_span = math.log(self.high) - log_low
        transformed = (torch.log(semantic) - log_low) / log_span
        transformed = torch.where(semantic == self.low, torch.zeros_like(transformed), transformed)
        transformed = torch.where(semantic == self.high, torch.ones_like(transformed), transformed)
        return transformed.clamp(0.0, 1.0).to(dtype=dtype)

    def encode_values(self, values: list[object], *, dtype: torch.dtype) -> torch.Tensor:
        validated = [self._validate_user_integer(value) for value in values]
        if self.log and not self.exponent:
            return self._encode_log_values(torch.tensor(validated, dtype=torch.int64), dtype=dtype)
        result = torch.tensor(validated, dtype=dtype)
        if self.exponent:
            transformed = torch.log(result) / math.log(self.base)
            lower, upper = self.optimization_bounds
            return transformed.clamp(lower, upper)
        if self.step != 1:
            return (result - self.low) / self.step
        return result

    def decode_values(self, values: torch.Tensor) -> tuple[object, ...]:
        coordinates = values.detach().to(device="cpu", dtype=torch.float64)
        if self.exponent:
            decoded = torch.pow(
                torch.tensor(self.base, dtype=torch.float64), coordinates.round()
            ).round()
        elif self.log:
            coordinates = coordinates.clamp(0.0, 1.0)
            log_low = math.log(self.low)
            log_span = math.log(self.high) - log_low
            decoded = torch.exp(log_low + coordinates * log_span).round()
        elif self.step != 1:
            decoded = coordinates.round() * self.step + self.low
        else:
            decoded = coordinates.round()
        decoded = decoded.clamp(self.low, self.high).to(torch.int64)
        return tuple(int(value) for value in decoded.tolist())

    def to_spec(self) -> dict[str, object]:
        return self._with_condition(
            {
                "name": self.name,
                "type": self.type_name,
                "low": self.low,
                "high": self.high,
                "step": self.step,
                "log": self.log,
                "base": self.base,
                "exponent": self.exponent,
            }
        )


@dataclass(frozen=True, slots=True, repr=False)
class Categorical(Parameter):
    """A finite categorical parameter stored as an integer code."""

    categories: tuple[Primitive, ...]
    _category_to_code: Mapping[Primitive, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    type_name: ClassVar[str] = "categorical"

    def __post_init__(self) -> None:
        Parameter.__post_init__(self)
        normalized: list[Primitive] = []
        for category in self.categories:
            if type(category).__module__.split(".", 1)[0] == "numpy" and hasattr(category, "item"):
                category = cast(Any, category).item()
            normalized.append(category)
        categories = tuple(normalized)
        if not categories:
            raise ValueError(f"categorical parameter {self.name!r} needs at least one category")
        for category in categories:
            if not isinstance(category, (str, int, float, bool, type(None))):
                raise TypeError(
                    "categories must be checkpoint-safe scalar primitives; "
                    f"got {type(category).__name__}"
                )
            if isinstance(category, float) and not math.isfinite(category):
                raise ValueError("floating categories must be finite")
        for index, category in enumerate(categories):
            if any(category == previous for previous in categories[:index]):
                raise ValueError(f"duplicate category {category!r} for {self.name!r}")
        object.__setattr__(self, "categories", categories)
        object.__setattr__(
            self,
            "_category_to_code",
            MappingProxyType({category: code for code, category in enumerate(categories)}),
        )

    @property
    def is_categorical(self) -> bool:
        return True

    @property
    def optimization_bounds(self) -> tuple[float, float]:
        return 0.0, float(len(self.categories) - 1)

    def _code(self, value: object) -> int:
        try:
            code = self._category_to_code.get(cast(Primitive, value))
        except TypeError as error:
            raise ValueError(f"unknown category {value!r} for {self.name!r}") from error
        if code is None:
            raise ValueError(f"unknown category {value!r} for {self.name!r}")
        return code

    def encode_values(self, values: list[object], *, dtype: torch.dtype) -> torch.Tensor:
        del dtype
        return torch.tensor([self._code(value) for value in values], dtype=torch.int64)

    def decode_values(self, values: torch.Tensor) -> tuple[object, ...]:
        codes = values.detach().to(device="cpu", dtype=torch.int64)
        if codes.numel() and ((codes < 0).any() or (codes >= len(self.categories)).any()):
            raise ValueError(f"categorical codes for {self.name!r} are outside the valid range")
        return tuple(self.categories[code] for code in codes.tolist())

    def to_spec(self) -> dict[str, object]:
        return self._with_condition(
            {"name": self.name, "type": self.type_name, "categories": list(self.categories)}
        )


@dataclass(frozen=True, slots=True, repr=False)
class Bool(Parameter):
    """A Boolean parameter stored in the categorical tensor as code 0 or 1."""

    type_name: ClassVar[str] = "bool"

    @property
    def is_categorical(self) -> bool:
        return True

    @property
    def optimization_bounds(self) -> tuple[float, float]:
        return 0.0, 1.0

    @property
    def is_discrete_after_transform(self) -> bool:
        return True

    def encode_values(self, values: list[object], *, dtype: torch.dtype) -> torch.Tensor:
        del dtype
        codes: list[int] = []
        for value in values:
            # NumPy Boolean scalars intentionally avoid a hard NumPy import.
            is_numpy_bool = type(value).__module__.split(".", 1)[0] == "numpy" and type(
                value
            ).__name__ in {"bool", "bool_"}
            if not isinstance(value, bool) and not is_numpy_bool:
                raise TypeError(f"{self.name!r} expects Boolean values")
            codes.append(1 if bool(value) else 0)
        return torch.tensor(codes, dtype=torch.int64)

    def decode_values(self, values: torch.Tensor) -> tuple[object, ...]:
        codes = values.detach().to(device="cpu", dtype=torch.int64)
        if codes.numel() and ((codes < 0).any() or (codes > 1).any()):
            raise ValueError(f"Boolean codes for {self.name!r} must be zero or one")
        return tuple(bool(code) for code in codes.tolist())

    def to_spec(self) -> dict[str, object]:
        return self._with_condition({"name": self.name, "type": self.type_name})


ParameterLike: TypeAlias = Float | Integer | Categorical | Bool


def parameter_from_spec(spec: dict[str, Any]) -> ParameterLike:
    """Construct one public parameter from a serialized specification."""

    item = dict(spec)
    try:
        type_name = str(item.pop("type"))
        name = str(item.pop("name"))
    except KeyError as error:
        raise ValueError(f"missing parameter-spec field: {error.args[0]}") from error
    active_spec = item.pop("active_when", None)
    if active_spec is not None and not isinstance(active_spec, Mapping):
        raise TypeError("active_when parameter specs must be mappings")
    active_when = None if active_spec is None else condition_from_spec(active_spec)
    if type_name == "float":
        return Float(name, active_when=active_when, **item)
    if type_name == "integer":
        return Integer(name, active_when=active_when, **item)
    if type_name == "categorical":
        return Categorical(name, active_when=active_when, **item)
    if type_name == "bool":
        if item:
            raise ValueError(f"unexpected Boolean parameter fields: {sorted(item)}")
        return Bool(name, active_when=active_when)
    raise ValueError(f"unsupported parameter type {type_name!r}")
