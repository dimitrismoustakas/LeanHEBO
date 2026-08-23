# SPDX-License-Identifier: MIT

"""Compiled tensor semantics for conditional design spaces."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch

from leanhebo.data.batch import EncodedBatch
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
from leanhebo.space.parameters import Bool, Categorical, Float, Integer, ParameterLike

_Comparison: TypeAlias = Literal["eq", "not_equal", "in", "lt", "le", "gt", "ge"]
_Storage: TypeAlias = Literal["continuous", "log_integer", "categorical", "static"]


def _is_static(parameter: ParameterLike) -> bool:
    return (isinstance(parameter, Integer) and parameter.low == parameter.high) or (
        isinstance(parameter, Categorical) and len(parameter.categories) == 1
    )


def _representative(parameter: ParameterLike) -> object:
    if isinstance(parameter, Float):
        return parameter.low
    if isinstance(parameter, Integer):
        return parameter.low
    if isinstance(parameter, Categorical):
        return parameter.categories[0]
    if isinstance(parameter, Bool):
        return False
    raise TypeError(f"unsupported parameter type {type(parameter).__name__}")


def _finite_values(parameter: ParameterLike) -> tuple[object, ...]:
    if isinstance(parameter, Integer):
        if parameter.exponent:
            lower, upper = parameter.optimization_bounds
            return tuple(
                int(parameter.base) ** exponent
                for exponent in range(round(lower), round(upper) + 1)
            )
        return tuple(range(parameter.low, parameter.high + 1, parameter.step))
    if isinstance(parameter, Categorical):
        return tuple(parameter.categories)
    if isinstance(parameter, Bool):
        return (False, True)
    if isinstance(parameter, Float):
        raise TypeError(f"parameter {parameter.name!r} has an infinite domain")
    raise TypeError(f"unsupported parameter type {type(parameter).__name__}")


def _numeric_threshold(parameter: Float | Integer, value: float | int) -> float:
    threshold = float(value)
    if isinstance(parameter, Float):
        if not parameter.log:
            return threshold
        if threshold <= 0:
            return -math.inf
        return math.log(threshold) / math.log(parameter.base)
    if parameter.exponent:
        if threshold <= 0:
            return -math.inf
        return math.log(threshold) / math.log(parameter.base)
    if parameter.log:
        if threshold <= 0:
            return -math.inf
        if parameter.low == parameter.high:
            return 0.0
        log_low = math.log(parameter.low)
        log_span = math.log(parameter.high) - log_low
        return (math.log(threshold) - log_low) / log_span
    if parameter.step != 1:
        return (threshold - parameter.low) / parameter.step
    return threshold


def _decode_log_integer(
    coordinate: torch.Tensor,
    *,
    semantic_low: int,
    semantic_high: int,
    log_low: float,
    log_span: float,
) -> torch.Tensor:
    return (
        torch.exp(log_low + coordinate.to(torch.float64).clamp(0.0, 1.0) * log_span)
        .round()
        .clamp(semantic_low, semantic_high)
        .to(torch.int64)
    )


@dataclass(frozen=True, slots=True)
class ActivityBatch:
    """Parameter- and group-level activity for one encoded batch."""

    parameter: torch.Tensor
    group: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CompiledAtom:
    parent_index: int
    storage: _Storage
    column: int
    comparison: _Comparison
    values: tuple[float | int, ...]
    static_result: bool = False
    semantic_low: int = 0
    semantic_high: int = 0
    log_low: float = 0.0
    log_span: float = 0.0

    def evaluate(
        self,
        encoded: EncodedBatch,
        activity: torch.Tensor,
        log_integer_values: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        guard = activity[:, self.parent_index]
        if self.storage == "static":
            return guard & self.static_result
        column = (
            encoded.continuous[:, self.column]
            if self.storage in {"continuous", "log_integer"}
            else encoded.categorical[:, self.column]
        )
        if self.storage == "log_integer":
            cached = log_integer_values.get(self.parent_index)
            if cached is None:
                cached = _decode_log_integer(
                    column,
                    semantic_low=self.semantic_low,
                    semantic_high=self.semantic_high,
                    log_low=self.log_low,
                    log_span=self.log_span,
                )
                log_integer_values[self.parent_index] = cached
            column = cached
        if self.comparison in {"eq", "not_equal", "in"}:
            matches = torch.zeros_like(guard)
            for value in self.values:
                matches |= column == value
            if self.comparison == "not_equal":
                matches = ~matches
            return guard & matches
        threshold = self.values[0]
        if self.comparison == "lt":
            result = column < threshold
        elif self.comparison == "le":
            result = column <= threshold
        elif self.comparison == "gt":
            result = column > threshold
        else:
            result = column >= threshold
        return guard & result


@dataclass(frozen=True, slots=True)
class _CompiledAll:
    children: tuple[_CompiledNode, ...]

    def evaluate(
        self,
        encoded: EncodedBatch,
        activity: torch.Tensor,
        log_integer_values: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        result = torch.ones(encoded.continuous.shape[0], dtype=torch.bool, device=encoded.device)
        for child in self.children:
            result &= child.evaluate(encoded, activity, log_integer_values)
        return result


@dataclass(frozen=True, slots=True)
class _CompiledAny:
    children: tuple[_CompiledNode, ...]

    def evaluate(
        self,
        encoded: EncodedBatch,
        activity: torch.Tensor,
        log_integer_values: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        result = torch.zeros(encoded.continuous.shape[0], dtype=torch.bool, device=encoded.device)
        for child in self.children:
            result |= child.evaluate(encoded, activity, log_integer_values)
        return result


_CompiledNode: TypeAlias = _CompiledAtom | _CompiledAll | _CompiledAny


class _ContextState:
    """One mutable CPU row evaluated by the compiled condition program."""

    def __init__(self, semantics: ConditionalSemantics) -> None:
        self.semantics = semantics
        self.encoded = EncodedBatch(
            semantics.continuous_representatives[None, :].clone(),
            semantics.categorical_representatives[None, :].clone(),
        )
        self.activity = torch.zeros((1, len(semantics.parameters)), dtype=torch.bool)
        self.log_integer_values: dict[int, torch.Tensor] = {}
        self.continuous_columns = {
            parameter_index: column
            for column, parameter_index in enumerate(semantics.continuous_parameter_indices)
        }
        self.categorical_columns = {
            parameter_index: column
            for column, parameter_index in enumerate(semantics.categorical_parameter_indices)
        }

    def is_active(self, parameter_index: int) -> bool:
        program = self.semantics.programs[parameter_index]
        if program is None:
            return True
        result = program.evaluate(self.encoded, self.activity, self.log_integer_values)
        return bool(result[0].item())

    def assign(self, parameter_index: int, value: object, *, active: bool) -> None:
        self.activity[0, parameter_index] = active
        self.log_integer_values.pop(parameter_index, None)
        parameter = self.semantics.parameters[parameter_index]
        if _is_static(parameter):
            return
        encoded = parameter.encode_values([value], dtype=self.semantics.dtype)
        continuous_column = self.continuous_columns.get(parameter_index)
        if continuous_column is not None:
            self.encoded.continuous[0, continuous_column] = encoded[0]
            return
        categorical_column = self.categorical_columns[parameter_index]
        self.encoded.categorical[0, categorical_column] = encoded[0].to(torch.int64)


@dataclass(frozen=True, slots=True)
class ConditionalSemantics:
    """Device-neutral compiled condition program for one parameter schema."""

    parameters: tuple[ParameterLike, ...]
    dtype: torch.dtype
    topological_order: tuple[int, ...]
    ancestor_parameter_indices: tuple[tuple[int, ...], ...]
    programs: tuple[_CompiledNode | None, ...]
    parameter_to_group: tuple[int, ...]
    group_parameter_indices: tuple[tuple[int, ...], ...]
    continuous_parameter_indices: tuple[int, ...]
    categorical_parameter_indices: tuple[int, ...]
    float_key_columns: tuple[int, ...]
    float_key_parameter_indices: tuple[int, ...]
    discrete_integer_key_columns: tuple[int, ...]
    discrete_integer_key_parameter_indices: tuple[int, ...]
    log_integer_key_columns: tuple[int, ...]
    log_integer_key_parameter_indices: tuple[int, ...]
    continuous_representatives: torch.Tensor
    categorical_representatives: torch.Tensor

    @classmethod
    def compile(
        cls,
        parameters: tuple[ParameterLike, ...],
        *,
        dtype: torch.dtype,
    ) -> ConditionalSemantics:
        names = {parameter.name: index for index, parameter in enumerate(parameters)}
        references: list[set[int]] = []
        adjacency: list[list[int]] = [[] for _ in parameters]
        indegree = [0] * len(parameters)
        for child, parameter in enumerate(parameters):
            condition = parameter.active_when
            child_references: set[int] = set()
            if condition is not None:
                unknown = sorted(condition.references.difference(names))
                if unknown:
                    raise ValueError(
                        f"condition for {parameter.name!r} references unknown parameters: {unknown}"
                    )
                child_references = {names[name] for name in condition.references}
            references.append(child_references)
            indegree[child] = len(child_references)
            for parent in child_references:
                adjacency[parent].append(child)

        ready = [index for index, degree in enumerate(indegree) if degree == 0]
        heapq.heapify(ready)
        topological: list[int] = []
        while ready:
            parent = heapq.heappop(ready)
            topological.append(parent)
            for child in sorted(adjacency[parent]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(topological) != len(parameters):
            cyclic = [parameters[index].name for index, degree in enumerate(indegree) if degree]
            raise ValueError(f"conditional parameter dependencies contain a cycle: {cyclic}")
        ancestor_sets: list[set[int]] = [set() for _ in parameters]
        for index in topological:
            ancestors = ancestor_sets[index]
            for parent in references[index]:
                ancestors.add(parent)
                ancestors.update(ancestor_sets[parent])
        topological_position = {index: position for position, index in enumerate(topological)}
        ancestor_parameter_indices = tuple(
            tuple(sorted(ancestors, key=topological_position.__getitem__))
            for ancestors in ancestor_sets
        )

        continuous_indices = tuple(
            index
            for index, parameter in enumerate(parameters)
            if not _is_static(parameter) and not parameter.is_categorical
        )
        categorical_indices = tuple(
            index
            for index, parameter in enumerate(parameters)
            if not _is_static(parameter) and parameter.is_categorical
        )
        continuous_columns = {index: column for column, index in enumerate(continuous_indices)}
        categorical_columns = {index: column for column, index in enumerate(categorical_indices)}
        float_key_indices: list[int] = []
        discrete_integer_key_indices: list[int] = []
        log_integer_key_indices: list[int] = []
        for index in continuous_indices:
            parameter = parameters[index]
            if isinstance(parameter, Float):
                float_key_indices.append(index)
            elif isinstance(parameter, Integer) and parameter.log and not parameter.exponent:
                log_integer_key_indices.append(index)
            elif isinstance(parameter, Integer):
                discrete_integer_key_indices.append(index)
            else:
                raise TypeError("compiled continuous-key metadata is inconsistent")
        float_key_parameter_indices = tuple(float_key_indices)
        discrete_integer_key_parameter_indices = tuple(discrete_integer_key_indices)
        log_integer_key_parameter_indices = tuple(log_integer_key_indices)

        def compile_condition(condition: Condition) -> _CompiledNode:
            if isinstance(condition, All):
                children = tuple(compile_condition(child) for child in condition.conditions)
                return _CompiledAll(children)
            if isinstance(condition, Any):
                children = tuple(compile_condition(child) for child in condition.conditions)
                return _CompiledAny(children)
            parent_index = names[next(iter(condition.references))]
            parameter = parameters[parent_index]
            if _is_static(parameter):
                storage: _Storage = "static"
                column = -1
            elif isinstance(parameter, Integer) and parameter.log and not parameter.exponent:
                storage = "log_integer"
                column = continuous_columns[parent_index]
            elif parameter.is_categorical:
                storage = "categorical"
                column = categorical_columns[parent_index]
            else:
                storage = "continuous"
                column = continuous_columns[parent_index]
            log_parameter = (
                parameter if storage == "log_integer" and isinstance(parameter, Integer) else None
            )
            semantic_low = 0 if log_parameter is None else log_parameter.low
            semantic_high = 0 if log_parameter is None else log_parameter.high
            log_low = 0.0 if log_parameter is None else math.log(log_parameter.low)
            log_span = (
                0.0
                if log_parameter is None
                else math.log(log_parameter.high) - math.log(log_parameter.low)
            )

            if isinstance(condition, (Eq, NotEqual, In)):
                if not isinstance(parameter, (Integer, Categorical, Bool)):
                    raise TypeError(
                        f"{type(condition).__name__} requires a discrete parameter; "
                        f"{parameter.name!r} is Float"
                    )
                raw_values = condition.values if isinstance(condition, In) else (condition.value,)
                encoded_values = parameter.encode_values(list(raw_values), dtype=dtype)
                values: tuple[float | int, ...]
                if log_parameter is not None:
                    values = tuple(
                        log_parameter._validate_user_integer(value) for value in raw_values
                    )
                else:
                    values = tuple(
                        int(value) if parameter.is_categorical else float(value)
                        for value in encoded_values.tolist()
                    )
                comparison: _Comparison
                if isinstance(condition, Eq):
                    comparison = "eq"
                elif isinstance(condition, NotEqual):
                    comparison = "not_equal"
                else:
                    comparison = "in"
                static_result = False
                if storage == "static":
                    static_encoded = parameter.encode_values(
                        [_representative(parameter)],
                        dtype=dtype,
                    )
                    static_value = (
                        int(static_encoded.item())
                        if parameter.is_categorical
                        else float(static_encoded.item())
                    )
                    matched = any(static_value == value for value in values)
                    static_result = not matched if comparison == "not_equal" else matched
                return _CompiledAtom(
                    parent_index,
                    storage,
                    column,
                    comparison,
                    values,
                    static_result,
                    semantic_low=semantic_low,
                    semantic_high=semantic_high,
                    log_low=log_low,
                    log_span=log_span,
                )

            if not isinstance(parameter, (Float, Integer)):
                raise TypeError(
                    f"{type(condition).__name__} requires a numeric parameter; "
                    f"{parameter.name!r} is categorical"
                )
            ordered_types = (LessThan, LessEqual, GreaterThan, GreaterEqual)
            if not isinstance(condition, ordered_types):
                raise TypeError(f"unsupported condition node {type(condition).__name__}")
            threshold = (
                float(condition.value)
                if storage == "log_integer"
                else _numeric_threshold(parameter, condition.value)
            )
            if isinstance(condition, LessThan):
                comparison = "lt"
            elif isinstance(condition, LessEqual):
                comparison = "le"
            elif isinstance(condition, GreaterThan):
                comparison = "gt"
            else:
                comparison = "ge"
            static_result = False
            if storage == "static":
                value = cast(float | int, _representative(parameter))
                if isinstance(condition, LessThan):
                    static_result = value < condition.value
                elif isinstance(condition, LessEqual):
                    static_result = value <= condition.value
                elif isinstance(condition, GreaterThan):
                    static_result = value > condition.value
                else:
                    static_result = value >= condition.value
            return _CompiledAtom(
                parent_index,
                storage,
                column,
                comparison,
                (threshold,),
                static_result,
                semantic_low=semantic_low,
                semantic_high=semantic_high,
                log_low=log_low,
                log_span=log_span,
            )

        program_by_condition: dict[Condition, _CompiledNode] = {}
        programs_list: list[_CompiledNode | None] = []
        for parameter in parameters:
            condition = parameter.active_when
            if condition is None:
                programs_list.append(None)
                continue
            program = program_by_condition.get(condition)
            if program is None:
                program = compile_condition(condition)
                program_by_condition[condition] = program
            programs_list.append(program)
        programs = tuple(programs_list)
        group_by_condition: dict[Condition, int] = {}
        grouped: list[list[int]] = []
        parameter_to_group: list[int] = []
        for index, parameter in enumerate(parameters):
            condition = parameter.active_when
            if condition is None:
                parameter_to_group.append(-1)
                continue
            group = group_by_condition.get(condition)
            if group is None:
                group = len(grouped)
                group_by_condition[condition] = group
                grouped.append([])
            grouped[group].append(index)
            parameter_to_group.append(group)

        continuous_representatives = torch.tensor(
            [parameters[index].optimization_bounds[0] for index in continuous_indices],
            dtype=dtype,
        )
        categorical_representatives = torch.tensor(
            [round(parameters[index].optimization_bounds[0]) for index in categorical_indices],
            dtype=torch.int64,
        )
        return cls(
            parameters,
            dtype,
            tuple(topological),
            ancestor_parameter_indices,
            programs,
            tuple(parameter_to_group),
            tuple(tuple(indices) for indices in grouped),
            continuous_indices,
            categorical_indices,
            tuple(continuous_columns[index] for index in float_key_parameter_indices),
            float_key_parameter_indices,
            tuple(continuous_columns[index] for index in discrete_integer_key_parameter_indices),
            discrete_integer_key_parameter_indices,
            tuple(continuous_columns[index] for index in log_integer_key_parameter_indices),
            log_integer_key_parameter_indices,
            continuous_representatives,
            categorical_representatives,
        )

    def _compiled_dtype(self, encoded: EncodedBatch) -> EncodedBatch:
        if encoded.dtype == self.dtype:
            return encoded
        return EncodedBatch(encoded.continuous.to(dtype=self.dtype), encoded.categorical)

    def _parameter_activity(
        self,
        encoded: EncodedBatch,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        row_count = len(encoded)
        parameter = torch.zeros(
            (row_count, len(self.parameters)),
            dtype=torch.bool,
            device=encoded.device,
        )
        program_results: dict[_CompiledNode, torch.Tensor] = {}
        log_integer_values: dict[int, torch.Tensor] = {}
        for index in self.topological_order:
            program = self.programs[index]
            if program is None:
                parameter[:, index] = True
                continue
            result = program_results.get(program)
            if result is None:
                result = program.evaluate(encoded, parameter, log_integer_values)
                program_results[program] = result
            parameter[:, index] = result
        return parameter, log_integer_values

    def activity(self, encoded: EncodedBatch) -> ActivityBatch:
        encoded = self._compiled_dtype(encoded)
        parameter, _ = self._parameter_activity(encoded)
        group = torch.stack(
            [parameter[:, indices[0]] for indices in self.group_parameter_indices],
            dim=1,
        )
        return ActivityBatch(parameter, group)

    def encode_records(self, records: list[dict[str, object]]) -> EncodedBatch:
        """Encode possibly sparse records in dependency order."""

        row_count = len(records)
        continuous = self.continuous_representatives.expand(row_count, -1).clone()
        categorical = self.categorical_representatives.expand(row_count, -1).clone()
        encoded = EncodedBatch(continuous, categorical)
        parameter_activity = torch.zeros(
            (row_count, len(self.parameters)),
            dtype=torch.bool,
        )
        continuous_columns = {
            parameter_index: column
            for column, parameter_index in enumerate(self.continuous_parameter_indices)
        }
        categorical_columns = {
            parameter_index: column
            for column, parameter_index in enumerate(self.categorical_parameter_indices)
        }
        program_results: dict[_CompiledNode, torch.Tensor] = {}
        log_integer_values: dict[int, torch.Tensor] = {}
        for index in self.topological_order:
            parameter = self.parameters[index]
            program = self.programs[index]
            if program is None:
                active = torch.ones(row_count, dtype=torch.bool)
            else:
                cached = program_results.get(program)
                if cached is None:
                    active = program.evaluate(
                        encoded,
                        parameter_activity,
                        log_integer_values,
                    )
                    program_results[program] = active
                else:
                    active = cached
            parameter_activity[:, index] = active
            missing = [
                row
                for row, record in enumerate(records)
                if bool(active[row]) and parameter.name not in record
            ]
            if missing:
                raise ValueError(
                    f"record {missing[0]} is missing active parameter {parameter.name!r}"
                )
            values = [
                record[parameter.name] if parameter.name in record else _representative(parameter)
                for record in records
            ]
            encoded_values = parameter.encode_values(values, dtype=self.dtype)
            if index in continuous_columns:
                continuous[:, continuous_columns[index]] = encoded_values
            elif index in categorical_columns:
                categorical[:, categorical_columns[index]] = encoded_values.to(torch.int64)
        return encoded

    def _project(
        self,
        encoded: EncodedBatch,
        parameter_activity: torch.Tensor,
    ) -> EncodedBatch:
        continuous_mask = parameter_activity[:, self.continuous_parameter_indices]
        categorical_mask = parameter_activity[:, self.categorical_parameter_indices]
        continuous = torch.where(
            continuous_mask,
            encoded.continuous,
            self.continuous_representatives.to(encoded.device, dtype=encoded.dtype),
        )
        categorical = torch.where(
            categorical_mask,
            encoded.categorical,
            self.categorical_representatives.to(encoded.device),
        )
        return EncodedBatch(continuous, categorical)

    def key_tensor(self, encoded: EncodedBatch) -> torch.Tensor:
        """Return exact semantic key components without leaving the input device."""

        encoded = self._compiled_dtype(encoded)
        activity, log_integer_values = self._parameter_activity(encoded)
        value_components: list[torch.Tensor] = []
        value_activity: list[torch.Tensor] = []

        if self.float_key_columns:
            values = encoded.continuous[:, self.float_key_columns]
            values = torch.where(values == 0, torch.zeros_like(values), values).contiguous()
            bits = (
                values.view(torch.int32).to(torch.int64)
                if self.dtype == torch.float32
                else values.view(torch.int64)
            )
            value_components.append(bits)
            value_activity.append(activity[:, self.float_key_parameter_indices])

        if self.discrete_integer_key_columns:
            values = encoded.continuous[:, self.discrete_integer_key_columns]
            value_components.append(values.round().to(torch.int64))
            value_activity.append(activity[:, self.discrete_integer_key_parameter_indices])

        for column, parameter_index in zip(
            self.log_integer_key_columns,
            self.log_integer_key_parameter_indices,
            strict=True,
        ):
            parameter = self.parameters[parameter_index]
            if not isinstance(parameter, Integer):
                raise TypeError("compiled log-integer key metadata is inconsistent")
            semantic = log_integer_values.get(parameter_index)
            if semantic is None:
                log_low = math.log(parameter.low)
                semantic = _decode_log_integer(
                    encoded.continuous[:, column],
                    semantic_low=parameter.low,
                    semantic_high=parameter.high,
                    log_low=log_low,
                    log_span=math.log(parameter.high) - log_low,
                )
                log_integer_values[parameter_index] = semantic
            value_components.append(semantic[:, None])
            value_activity.append(activity[:, parameter_index, None])

        if self.categorical_parameter_indices:
            value_components.append(encoded.categorical.to(torch.int64))
            value_activity.append(activity[:, self.categorical_parameter_indices])

        if not value_components:
            return activity.to(torch.int64)
        values = torch.cat(value_components, dim=1)
        active_values = torch.cat(value_activity, dim=1)
        masked_values = torch.where(active_values, values, torch.zeros_like(values))
        return torch.cat((activity.to(torch.int64), masked_values), dim=1)

    def context_is_finite(self, fixed_values: Mapping[str, object]) -> bool:
        """Decide whether every unfixed floating parameter is necessarily inactive."""

        if all(
            not isinstance(parameter, Float) or parameter.name in fixed_values
            for parameter in self.parameters
        ):
            return True

        def can_unfixed_float_activate(target_index: int) -> bool:
            target = self.parameters[target_index]
            if target.active_when is None:
                return True
            relevant = self.ancestor_parameter_indices[target_index]
            state = _ContextState(self)

            def can_activate(position: int) -> bool:
                if position == len(relevant):
                    return state.is_active(target_index)
                parameter_index = relevant[position]
                parameter = self.parameters[parameter_index]
                active = state.is_active(parameter_index)
                if not active:
                    state.assign(
                        parameter_index,
                        _representative(parameter),
                        active=False,
                    )
                    return can_activate(position + 1)
                if parameter.name in fixed_values:
                    state.assign(
                        parameter_index,
                        fixed_values[parameter.name],
                        active=True,
                    )
                    return can_activate(position + 1)
                if isinstance(parameter, Float):
                    return True
                for value in _finite_values(parameter):
                    state.assign(parameter_index, value, active=True)
                    if can_activate(position + 1):
                        return True
                return False

            return can_activate(0)

        for target_index, target in enumerate(self.parameters):
            if (
                isinstance(target, Float)
                and target.name not in fixed_values
                and can_unfixed_float_activate(target_index)
            ):
                return False
        return True

    def iter_contextual_records(
        self,
        fixed_values: Mapping[str, object],
    ) -> Iterator[dict[str, object]]:
        if not self.context_is_finite(fixed_values):
            raise ValueError("the contextual semantic domain is infinite")
        yield from self._iter_contextual_records(fixed_values)

    def _iter_contextual_records(
        self,
        fixed_values: Mapping[str, object],
    ) -> Iterator[dict[str, object]]:
        """Yield records after the caller has established contextual finiteness."""

        values: dict[str, object] = {}
        state = _ContextState(self)

        def generate(position: int) -> Iterator[dict[str, object]]:
            if position == len(self.topological_order):
                yield {
                    parameter.name: values[parameter.name]
                    for index, parameter in enumerate(self.parameters)
                    if bool(state.activity[0, index].item())
                }
                return
            parameter_index = self.topological_order[position]
            parameter = self.parameters[parameter_index]
            active = state.is_active(parameter_index)
            if not active:
                value = _representative(parameter)
                values[parameter.name] = value
                state.assign(parameter_index, value, active=False)
                yield from generate(position + 1)
                return
            choices = (
                (fixed_values[parameter.name],)
                if parameter.name in fixed_values
                else _finite_values(parameter)
            )
            for value in choices:
                values[parameter.name] = value
                state.assign(parameter_index, value, active=True)
                yield from generate(position + 1)

        yield from generate(0)


__all__ = ["ConditionalSemantics"]
