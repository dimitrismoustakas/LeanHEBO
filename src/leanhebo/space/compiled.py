# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Immutable, tensor-native compilation of a public :class:`Space`."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, TypeAlias, cast

import torch

from leanhebo.data.adapters import columns_from_input
from leanhebo.data.batch import CandidateBatch, EncodedBatch
from leanhebo.errors import SpaceMismatchError
from leanhebo.space.parameters import Bool, Categorical, Float, Integer, ParameterLike

ExternalBatch: TypeAlias = object


def _is_static(parameter: ParameterLike) -> bool:
    return (isinstance(parameter, Integer) and parameter.low == parameter.high) or (
        isinstance(parameter, Categorical) and len(parameter.categories) == 1
    )


def _static_value(parameter: ParameterLike) -> object:
    if isinstance(parameter, Integer) and parameter.low == parameter.high:
        return parameter.low
    if isinstance(parameter, Categorical) and len(parameter.categories) == 1:
        return parameter.categories[0]
    raise ValueError(f"parameter {parameter.name!r} is not static")


def _dtype_from_value(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, str):
        normalized = dtype.removeprefix("torch.")
        mapping = {"float32": torch.float32, "float64": torch.float64}
        try:
            dtype = mapping[normalized]
        except KeyError as error:
            raise ValueError("compiled spaces support dtype 'float32' or 'float64'") from error
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("compiled spaces support torch.float32 or torch.float64")
    return dtype


def _fingerprint(parameters: tuple[ParameterLike, ...]) -> str:
    payload = json.dumps(
        [parameter.to_spec() for parameter in parameters],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class FixedInput:
    """Pre-encoded contextual assignments for repeated search-time repair."""

    continuous_indices: torch.Tensor
    continuous_values: torch.Tensor
    categorical_indices: torch.Tensor
    categorical_values: torch.Tensor
    dense_mask: torch.Tensor
    space_fingerprint: str
    decoded_values: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.continuous_indices.dtype != torch.int64:
            raise TypeError("continuous fixed indices must use torch.int64")
        if self.categorical_indices.dtype != torch.int64:
            raise TypeError("categorical fixed indices must use torch.int64")
        if self.categorical_values.dtype != torch.int64:
            raise TypeError("categorical fixed values must use torch.int64")
        if not self.continuous_values.is_floating_point():
            raise TypeError("continuous fixed values must use a floating dtype")
        if self.dense_mask.dtype != torch.bool or self.dense_mask.ndim != 1:
            raise TypeError("dense fixed mask must be a one-dimensional Boolean tensor")
        if self.continuous_indices.numel() != self.continuous_values.numel():
            raise ValueError("continuous fixed indices and values have different lengths")
        if self.categorical_indices.numel() != self.categorical_values.numel():
            raise ValueError("categorical fixed indices and values have different lengths")

    def __len__(self) -> int:
        return self.continuous_indices.numel() + self.categorical_indices.numel()

    def to(
        self,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> FixedInput:
        target_dtype = self.continuous_values.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise TypeError("continuous fixed values require a floating dtype")
        return FixedInput(
            self.continuous_indices.to(device=device),
            self.continuous_values.to(device=device, dtype=target_dtype),
            self.categorical_indices.to(device=device),
            self.categorical_values.to(device=device),
            self.dense_mask.to(device=device),
            self.space_fingerprint,
            self.decoded_values,
        )


@dataclass(frozen=True, slots=True, eq=False)
class CompiledSpace:
    """Device-neutral metadata and vectorized codecs for one design schema."""

    parameters: tuple[ParameterLike, ...]
    dtype: torch.dtype = torch.float32
    names: tuple[str, ...] = field(init=False)
    fingerprint: str = field(init=False)
    dense_lower_bounds: torch.Tensor = field(init=False, repr=False)
    dense_upper_bounds: torch.Tensor = field(init=False, repr=False)
    categorical_mask: torch.Tensor = field(init=False, repr=False)
    rounding_mask: torch.Tensor = field(init=False, repr=False)
    _continuous_parameters: tuple[Float | Integer, ...] = field(init=False, repr=False)
    _categorical_parameters: tuple[Categorical | Bool, ...] = field(init=False, repr=False)
    _static_parameters: tuple[Integer | Categorical, ...] = field(init=False, repr=False)
    _name_positions: dict[str, tuple[Literal["continuous", "categorical"], int]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        if not parameters:
            raise ValueError("a compiled design space needs at least one parameter")
        names = tuple(parameter.name for parameter in parameters)
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate parameter names: {duplicates}")
        dtype = _dtype_from_value(self.dtype)
        active = tuple(parameter for parameter in parameters if not _is_static(parameter))
        continuous = tuple(parameter for parameter in active if not parameter.is_categorical)
        categorical = tuple(parameter for parameter in active if parameter.is_categorical)
        static = tuple(parameter for parameter in parameters if _is_static(parameter))
        dense_parameters = continuous + categorical
        continuous_bounds = [parameter.optimization_bounds for parameter in continuous]
        categorical_bounds = [parameter.optimization_bounds for parameter in categorical]
        continuous_lower = torch.tensor([bounds[0] for bounds in continuous_bounds], dtype=dtype)
        continuous_upper = torch.tensor([bounds[1] for bounds in continuous_bounds], dtype=dtype)
        categorical_lower = torch.tensor(
            [int(bounds[0]) for bounds in categorical_bounds], dtype=torch.int64
        )
        categorical_upper = torch.tensor(
            [int(bounds[1]) for bounds in categorical_bounds], dtype=torch.int64
        )
        categorical_mask = torch.tensor(
            [parameter.is_categorical for parameter in dense_parameters], dtype=torch.bool
        )
        rounding_mask = torch.tensor(
            [
                parameter.is_categorical
                or (isinstance(parameter, Integer) and parameter.mode != "power")
                for parameter in dense_parameters
            ],
            dtype=torch.bool,
        )
        positions: dict[str, tuple[Literal["continuous", "categorical"], int]] = {}
        for index, parameter in enumerate(continuous):
            positions[parameter.name] = ("continuous", index)
        for index, parameter in enumerate(categorical):
            positions[parameter.name] = ("categorical", index)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "fingerprint", _fingerprint(parameters))
        object.__setattr__(self, "_continuous_parameters", continuous)
        object.__setattr__(self, "_categorical_parameters", categorical)
        object.__setattr__(self, "_static_parameters", static)
        object.__setattr__(self, "_name_positions", positions)
        object.__setattr__(
            self,
            "dense_lower_bounds",
            torch.cat((continuous_lower, categorical_lower.to(dtype=dtype))),
        )
        object.__setattr__(
            self,
            "dense_upper_bounds",
            torch.cat((continuous_upper, categorical_upper.to(dtype=dtype))),
        )
        object.__setattr__(self, "categorical_mask", categorical_mask)
        object.__setattr__(self, "rounding_mask", rounding_mask)

    def __len__(self) -> int:
        return len(self.parameters)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CompiledSpace)
            and self.fingerprint == other.fingerprint
            and self.dtype == other.dtype
        )

    def __hash__(self) -> int:
        return hash((self.fingerprint, self.dtype))

    @property
    def n_continuous(self) -> int:
        return len(self._continuous_parameters)

    @property
    def n_categorical(self) -> int:
        return len(self._categorical_parameters)

    @property
    def categorical_parameters(self) -> tuple[Categorical | Bool, ...]:
        return self._categorical_parameters

    @property
    def dense_dimension(self) -> int:
        return self.n_continuous + self.n_categorical

    def to_spec(self) -> list[dict[str, object]]:
        return [parameter.to_spec() for parameter in self.parameters]

    def encode(self, value: ExternalBatch) -> EncodedBatch:
        """Encode supported boundary data exactly once."""

        if isinstance(value, CandidateBatch):
            self._validate_fingerprint(value.space_fingerprint)
            self.validate_encoded(value.encoded)
            return value.encoded
        if isinstance(value, EncodedBatch):
            self.validate_encoded(value)
            return value
        columns = columns_from_input(value, self.names)
        row_count = len(next(iter(columns.values())))
        for parameter in self._static_parameters:
            parameter.encode_values(columns[parameter.name], dtype=self.dtype)
        continuous_columns = [
            parameter.encode_values(columns[parameter.name], dtype=self.dtype)
            for parameter in self._continuous_parameters
        ]
        categorical_columns = [
            parameter.encode_values(columns[parameter.name], dtype=self.dtype)
            for parameter in self._categorical_parameters
        ]
        continuous = (
            torch.stack(continuous_columns, dim=1)
            if continuous_columns
            else torch.empty((row_count, 0), dtype=self.dtype)
        )
        categorical = (
            torch.stack(categorical_columns, dim=1).to(torch.int64)
            if categorical_columns
            else torch.empty((row_count, 0), dtype=torch.int64)
        )
        return EncodedBatch(continuous, categorical)

    def decode(
        self,
        value: EncodedBatch | CandidateBatch,
        *,
        fixed: FixedInput | Mapping[str, object] | None = None,
    ) -> CandidateBatch:
        """Decode tensors once and retain those same tensors in CandidateBatch."""

        if isinstance(value, CandidateBatch):
            self._validate_fingerprint(value.space_fingerprint)
            if value.decoded_columns is not None and fixed is None:
                return value
            encoded = value.encoded
        else:
            encoded = value
        self.validate_encoded(encoded)
        fixed_input = self._coerce_fixed(fixed)
        columns: dict[str, tuple[object, ...]] = {
            parameter.name: (_static_value(parameter),) * len(encoded)
            for parameter in self._static_parameters
        }
        for index, cont_parameter in enumerate(self._continuous_parameters):
            columns[cont_parameter.name] = cont_parameter.decode_values(
                encoded.continuous[:, index]
            )
        for index, cat_parameter in enumerate(self._categorical_parameters):
            columns[cat_parameter.name] = cat_parameter.decode_values(encoded.categorical[:, index])
        if fixed_input is not None:
            for name, fixed_value in fixed_input.decoded_values:
                columns[name] = (fixed_value,) * len(encoded)
        ordered = {name: columns[name] for name in self.names}
        return CandidateBatch(encoded.continuous, encoded.categorical, self.fingerprint, ordered)

    def validate_encoded(self, encoded: EncodedBatch) -> None:
        self._validate_shape(encoded)
        if not torch.isfinite(encoded.continuous).all():
            raise ValueError("encoded continuous coordinates must be finite")
        continuous_lower = self.dense_lower_bounds[: self.n_continuous].to(device=encoded.device)
        continuous_upper = self.dense_upper_bounds[: self.n_continuous].to(device=encoded.device)
        if encoded.continuous.numel() and (
            (encoded.continuous < continuous_lower).any()
            or (encoded.continuous > continuous_upper).any()
        ):
            raise ValueError("encoded continuous values exceed their bounds")
        for index, cont_parameter in enumerate(self._continuous_parameters):
            values = encoded.continuous[:, index]
            if (
                cont_parameter.is_discrete_after_transform
                and values.numel()
                and not torch.equal(values, values.round())
            ):
                raise ValueError(
                    f"encoded values for discrete parameter {cont_parameter.name!r} "
                    "are not integral"
                )
        categorical_lower = self.dense_lower_bounds[self.n_continuous :].to(
            device=encoded.device, dtype=torch.int64
        )
        categorical_upper = self.dense_upper_bounds[self.n_continuous :].to(
            device=encoded.device, dtype=torch.int64
        )
        if encoded.categorical.numel() and (
            (encoded.categorical < categorical_lower).any()
            or (encoded.categorical > categorical_upper).any()
        ):
            raise ValueError("encoded categorical values exceed their code ranges")

    def _validate_shape(self, encoded: EncodedBatch) -> None:
        if encoded.n_continuous != self.n_continuous:
            raise ValueError(
                f"expected {self.n_continuous} continuous columns, got {encoded.n_continuous}"
            )
        if encoded.n_categorical != self.n_categorical:
            raise ValueError(
                f"expected {self.n_categorical} categorical columns, got {encoded.n_categorical}"
            )

    def _validate_fingerprint(self, fingerprint: str) -> None:
        if fingerprint != self.fingerprint:
            raise SpaceMismatchError("CandidateBatch was produced by a different design space")

    def to_dense(self, value: EncodedBatch | CandidateBatch) -> torch.Tensor:
        encoded = self.encode(value)
        return encoded.to_dense()

    def encoded_from_dense(
        self,
        dense: torch.Tensor,
        *,
        repair: bool = True,
        fixed: FixedInput | Mapping[str, object] | None = None,
    ) -> EncodedBatch:
        """Split a dense search population into the native two-tensor layout."""

        if not isinstance(dense, torch.Tensor) or not dense.is_floating_point():
            raise TypeError("dense optimization coordinates must be a floating tensor")
        if dense.ndim != 2 or dense.shape[1] != self.dense_dimension:
            raise ValueError(f"dense coordinates must have shape [rows, {self.dense_dimension}]")
        values = dense.to(dtype=self.dtype)
        continuous = values[:, : self.n_continuous]
        categorical_values = values[:, self.n_continuous :]
        if repair:
            continuous = continuous.clone()
            categorical_values = categorical_values.clone()
            for index, cont_parameter in enumerate(self._continuous_parameters):
                lower = self.dense_lower_bounds[index].to(device=continuous.device)
                upper = self.dense_upper_bounds[index].to(device=continuous.device)
                column = continuous[:, index].clamp(lower, upper)
                if (
                    isinstance(cont_parameter, Integer)
                    and cont_parameter.log
                    and not cont_parameter.exponent
                ):
                    log_low = math.log(cont_parameter.low)
                    log_span = math.log(cont_parameter.high) - log_low
                    semantic = (
                        torch.exp(log_low + column.to(torch.float64) * log_span)
                        .round()
                        .clamp(cont_parameter.low, cont_parameter.high)
                    )
                    column = cont_parameter._encode_log_values(semantic, dtype=column.dtype).clamp(
                        lower, upper
                    )
                elif cont_parameter.is_discrete_after_transform:
                    column = column.round()
                continuous[:, index] = column
            for index in range(self.n_categorical):
                dense_index = self.n_continuous + index
                lower = self.dense_lower_bounds[dense_index].to(device=categorical_values.device)
                upper = self.dense_upper_bounds[dense_index].to(device=categorical_values.device)
                categorical_values[:, index] = (
                    categorical_values[:, index].round().clamp(lower, upper)
                )
        categorical = categorical_values.to(dtype=torch.int64)
        encoded = EncodedBatch(continuous, categorical)
        if not repair:
            self.validate_encoded(encoded)
        fixed_input = self._coerce_fixed(fixed)
        return self.apply_fixed(encoded, fixed_input) if fixed_input is not None else encoded

    def candidate_from_dense(
        self,
        dense: torch.Tensor,
        *,
        repair: bool = True,
        fixed: FixedInput | Mapping[str, object] | None = None,
    ) -> CandidateBatch:
        fixed_input = self._coerce_fixed(fixed)
        encoded = self.encoded_from_dense(dense, repair=repair, fixed=fixed_input)
        return self.decode(encoded, fixed=fixed_input)

    def compile_fixed(self, values: Mapping[str, object] | None = None) -> FixedInput:
        """Validate and encode a reusable contextual fixed-input assignment."""

        assignments = {} if values is None else dict(values)
        unknown = sorted(set(assignments).difference(self.names))
        if unknown:
            raise ValueError(f"unknown fixed-input parameters: {unknown}")
        continuous_indices: list[int] = []
        continuous_values: list[float] = []
        categorical_indices: list[int] = []
        categorical_values: list[int] = []
        decoded_values: list[tuple[str, object]] = []
        dense_mask = torch.zeros(self.dense_dimension, dtype=torch.bool)
        by_name = {parameter.name: parameter for parameter in self.parameters}
        for name in self.names:
            if name not in assignments:
                continue
            parameter = by_name[name]
            encoded = parameter.encode_values([assignments[name]], dtype=self.dtype)
            if _is_static(parameter):
                continue
            location, index = self._name_positions[name]
            if location == "continuous":
                continuous_indices.append(index)
                continuous_values.append(float(encoded.item()))
                dense_mask[index] = True
            else:
                categorical_indices.append(index)
                categorical_values.append(int(encoded.item()))
                dense_mask[self.n_continuous + index] = True
            decoded = parameter.decode_values(encoded)[0]
            # Logarithmic floating inversion need not be bit exact.  Retaining the
            # normalized boundary value guarantees contextual output exactness.
            if isinstance(parameter, Float):
                decoded = float(cast(Real, assignments[name]))
            decoded_values.append((name, decoded))
        return FixedInput(
            torch.tensor(continuous_indices, dtype=torch.int64),
            torch.tensor(continuous_values, dtype=self.dtype),
            torch.tensor(categorical_indices, dtype=torch.int64),
            torch.tensor(categorical_values, dtype=torch.int64),
            dense_mask,
            self.fingerprint,
            tuple(decoded_values),
        )

    def _coerce_fixed(self, fixed: FixedInput | Mapping[str, object] | None) -> FixedInput | None:
        if fixed is None:
            return None
        fixed_input = self.compile_fixed(fixed) if isinstance(fixed, Mapping) else fixed
        self._validate_fingerprint(fixed_input.space_fingerprint)
        if fixed_input.dense_mask.numel() != self.dense_dimension:
            raise ValueError("FixedInput has an incompatible dense dimension")
        return fixed_input

    def apply_fixed(self, encoded: EncodedBatch, fixed: FixedInput) -> EncodedBatch:
        self._validate_fingerprint(fixed.space_fingerprint)
        self._validate_shape(encoded)
        if len(fixed) == 0:
            return encoded
        continuous = encoded.continuous.clone()
        categorical = encoded.categorical.clone()
        if fixed.continuous_indices.numel():
            indices = fixed.continuous_indices.to(device=encoded.device)
            values = fixed.continuous_values.to(device=encoded.device, dtype=encoded.dtype)
            continuous[:, indices] = values
        if fixed.categorical_indices.numel():
            indices = fixed.categorical_indices.to(device=encoded.device)
            values = fixed.categorical_values.to(device=encoded.device)
            categorical[:, indices] = values
        return EncodedBatch(continuous, categorical)

    def sample(
        self,
        count: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        scramble: bool = True,
        fixed: FixedInput | Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> CandidateBatch:
        """Draw a mixed-space Sobol batch and decode it once."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("sample count must be a non-negative integer")
        if seed is not None and generator is not None:
            raise ValueError("pass either seed or generator, not both")
        if generator is not None:
            random_device = generator.device
            seed = int(
                torch.randint(
                    0,
                    2**31 - 1,
                    (1,),
                    generator=generator,
                    device=random_device,
                ).item()
            )
        if count == 0 or self.dense_dimension == 0:
            unit = torch.empty((count, self.dense_dimension), dtype=self.dtype)
        else:
            engine = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
                self.dense_dimension, scramble=scramble, seed=seed
            )
            unit = engine.draw(count, dtype=self.dtype)
        dense = torch.empty_like(unit)
        dense_parameters = self._continuous_parameters + self._categorical_parameters
        for index, parameter in enumerate(dense_parameters):
            lower, upper = parameter.optimization_bounds
            if parameter.is_discrete_after_transform:
                cardinality = round(upper - lower) + 1
                dense[:, index] = (
                    torch.floor(unit[:, index] * cardinality).clamp_max(cardinality - 1) + lower
                )
            else:
                dense[:, index] = lower + unit[:, index] * (upper - lower)
        if device is not None:
            dense = dense.to(device=device)
        fixed_input = self._coerce_fixed(fixed)
        return self.candidate_from_dense(dense, fixed=fixed_input)

    def canonical_key_tensor(self, value: EncodedBatch | CandidateBatch) -> torch.Tensor:
        """Return exact int64 key components in public schema order on CPU."""

        if isinstance(value, CandidateBatch):
            self._validate_fingerprint(value.space_fingerprint)
            encoded = value.encoded
        else:
            encoded = value
        self.validate_encoded(encoded)
        continuous = encoded.continuous.detach().to(device="cpu", dtype=self.dtype)
        categorical = encoded.categorical.detach().to(device="cpu")
        columns: dict[str, torch.Tensor] = {}
        for parameter in self._static_parameters:
            if isinstance(parameter, Integer):
                columns[parameter.name] = torch.full(
                    (len(encoded),), parameter.low, dtype=torch.int64
                )
            else:
                columns[parameter.name] = torch.zeros(len(encoded), dtype=torch.int64)
        for index, cont_parameter in enumerate(self._continuous_parameters):
            column = continuous[:, index].contiguous()
            if isinstance(cont_parameter, Float):
                column = torch.where(column == 0, torch.zeros_like(column), column)
                bits = (
                    column.view(torch.int32).to(torch.int64)
                    if self.dtype == torch.float32
                    else column.view(torch.int64)
                )
                columns[cont_parameter.name] = bits
            elif cont_parameter.log and not cont_parameter.exponent:
                decoded = cont_parameter.decode_values(column)
                columns[cont_parameter.name] = torch.tensor(decoded, dtype=torch.int64)
            else:
                columns[cont_parameter.name] = column.round().to(torch.int64)
        for index, cat_parameter in enumerate(self._categorical_parameters):
            columns[cat_parameter.name] = categorical[:, index].to(torch.int64)
        return torch.stack([columns[name] for name in self.names], dim=1)

    def canonical_keys(self, value: EncodedBatch | CandidateBatch) -> tuple[tuple[int, ...], ...]:
        tensor = self.canonical_key_tensor(value)
        return tuple(tuple(int(component) for component in row) for row in tensor.tolist())
