# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import pytest
import torch

import leanhebo.space.conditional as conditional_module
from leanhebo.data import EncodedBatch
from leanhebo.space import (
    All,
    Any,
    Bool,
    Categorical,
    CompiledSpace,
    Eq,
    Float,
    GreaterThan,
    In,
    Integer,
    LessEqual,
    NotEqual,
    Space,
)
from leanhebo.space.parameters import ParameterLike


def _semantics(space: CompiledSpace) -> conditional_module.ConditionalSemantics:
    semantics = space.conditional_semantics
    assert semantics is not None
    return semantics


def xgboost_space() -> Space:
    return Space(
        Float("rate_drop", 0.0, 1.0, active_when=Eq("booster", "dart")),
        Categorical("booster", ("gblinear", "gbtree", "dart")),
        Integer(
            "max_depth",
            1,
            4,
            active_when=In("booster", ("gbtree", "dart")),
        ),
    )


def test_condition_normalization_spec_round_trip_and_grouping() -> None:
    first = All(Eq("left", True), NotEqual("right", "off"))
    second = All(NotEqual("right", "off"), Eq("left", True))
    conditional = Space(
        Bool("left"),
        Categorical("right", ("off", "on")),
        Integer("a", 0, 2, active_when=first),
        Integer("b", 0, 2, active_when=second),
        Integer("c", 0, 2, active_when=In("left", (True,))),
        Integer("d", 0, 2, active_when=Eq("left", True)),
    )
    compiled = conditional.compile()
    semantics = compiled.conditional_semantics

    assert semantics is not None
    assert semantics.parameter_to_group == (-1, -1, 0, 0, 1, 1)
    assert semantics.programs[2] is semantics.programs[3]
    assert semantics.programs[4] is semantics.programs[5]
    assert conditional.parameters[4].active_when == Eq("left", True)
    assert Integer("x", 0, 2, active_when=All(In("left", (True,)))).active_when == Eq("left", True)
    assert "active_when=All(" in repr(conditional.parameters[2])

    restored = Space.from_spec(conditional.to_spec())
    assert restored == conditional
    assert restored.compile().fingerprint == compiled.fingerprint

    singleton_in = Space(
        Bool("left"),
        Integer("value", 0, 2, active_when=In("left", (True,))),
    ).compile()
    equality = Space(
        Bool("left"),
        Integer("value", 0, 2, active_when=Eq("left", True)),
    ).compile()
    assert singleton_in.to_spec() == equality.to_spec()
    assert singleton_in.fingerprint == equality.fingerprint

    numeric_first = Space(
        Integer("parent", 0, 1),
        Bool("child", active_when=All(Eq("parent", 0), Eq("parent", -0.0))),
    ).compile()
    numeric_reversed = Space(
        Integer("parent", 0, 1),
        Bool("child", active_when=All(Eq("parent", 0.0), Eq("parent", 0))),
    ).compile()
    assert In("parent", (0.0, -0.0, 0)).values == (0,)
    assert numeric_first.to_spec() == numeric_reversed.to_spec()
    assert numeric_first.fingerprint == numeric_reversed.fingerprint

    assert Eq("left", True) != Eq("left", 1)
    typed_values = In("left", (True, 1)).values
    assert tuple(type(value) for value in typed_values) == (int, bool)


@pytest.mark.parametrize("reverse", [False, True])
def test_boolean_conditions_do_not_hash_cons_invalid_integer_values(reverse: bool) -> None:
    children = (
        Bool("valid", active_when=Eq("parent", True)),
        Bool("invalid", active_when=Eq("parent", 1)),
    )
    if reverse:
        children = children[::-1]
    with pytest.raises(TypeError, match="expects Boolean values"):
        Space(Bool("parent"), *children).compile()

    operands = (Eq("parent", True), Eq("parent", 1))
    if reverse:
        operands = operands[::-1]
    with pytest.raises(TypeError, match="expects Boolean values"):
        Space(
            Bool("parent"),
            Bool("child", active_when=All(*operands)),
        ).compile()


def test_compilation_rejects_invalid_references_cycles_and_predicate_domains() -> None:
    with pytest.raises(ValueError, match="unknown parameters"):
        Space(Integer("x", 0, 2, active_when=Eq("missing", 1))).compile()

    with pytest.raises(ValueError, match="cycle"):
        Space(
            Bool("a", active_when=Eq("b", True)),
            Bool("b", active_when=Eq("a", True)),
        ).compile()

    with pytest.raises(TypeError, match="discrete parameter"):
        Space(Float("x", 0.0, 1.0), Bool("child", active_when=Eq("x", 0.5))).compile()

    with pytest.raises(TypeError, match="numeric parameter"):
        Space(
            Categorical("choice", ("a", "b")),
            Bool("child", active_when=LessEqual("choice", 1)),
        ).compile()

    with pytest.raises(ValueError, match="unknown category"):
        Space(
            Categorical("choice", ("a", "b")),
            Bool("child", active_when=Eq("choice", "c")),
        ).compile()


def test_atoms_are_guarded_when_any_references_inactive_parents() -> None:
    compiled = Space(
        Categorical("branch", ("left", "right")),
        Bool("left_gate", active_when=Eq("branch", "left")),
        Bool("right_gate", active_when=Eq("branch", "right")),
        Integer(
            "child",
            0,
            3,
            active_when=Any(Eq("left_gate", True), Eq("right_gate", True)),
        ),
    ).compile()
    records = [
        {
            "branch": "left",
            "left_gate": False,
            "right_gate": True,
        },
        {
            "branch": "left",
            "left_gate": True,
            "right_gate": True,
            "child": 2,
        },
        {
            "branch": "right",
            "left_gate": True,
            "right_gate": True,
            "child": 3,
        },
    ]

    candidates = compiled.decode(compiled.encode(records))

    assert candidates.activity is not None
    assert candidates.activity.tolist() == [
        [True, True, False, False],
        [True, True, False, True],
        [True, False, True, True],
    ]
    assert candidates.to_records() == [
        {"branch": "left", "left_gate": False},
        {"branch": "left", "left_gate": True, "child": 2},
        {"branch": "right", "right_gate": True, "child": 3},
    ]


def test_sparse_boundary_retains_latent_genes_but_projects_inactive_values() -> None:
    compiled = xgboost_space().compile(dtype=torch.float64)
    latent = compiled.encode([{"booster": "gblinear", "rate_drop": 0.9, "max_depth": 4}])
    omitted = compiled.encode([{"booster": "gblinear"}])

    assert not torch.equal(latent.continuous, omitted.continuous)
    assert compiled.canonical_keys(latent) == compiled.canonical_keys(omitted)

    candidates = compiled.decode(latent)
    assert candidates.to_records() == [{"booster": "gblinear"}]
    assert candidates.to_numpy().tolist() == [[0.0, "gblinear", 1]]
    assert candidates.activity is not None
    assert candidates.activity.tolist() == [[False, True, False]]

    moved = candidates.to(dtype=torch.float32)
    selected = moved.select([0])
    assert selected.activity is not None
    assert selected.activity.tolist() == [[False, True, False]]
    assert selected.to_records() == [{"booster": "gblinear"}]

    with pytest.raises(ValueError, match="missing active parameter 'max_depth'"):
        compiled.encode([{"booster": "gbtree"}])
    with pytest.raises(ValueError, match="outside"):
        compiled.encode([{"booster": "gblinear", "max_depth": 99}])
    with pytest.raises(ValueError, match="unknown columns"):
        compiled.encode([{"booster": "gblinear", "typo": 1}])
    with pytest.raises(ValueError, match="unknown columns"):
        compiled.encode({"booster": ["gblinear"], "typo": [1]})


def test_semantic_keys_include_activity_and_remain_on_the_input_device() -> None:
    compiled = Space(
        Bool("enabled"),
        Integer("child", 0, 2, active_when=Eq("enabled", True)),
    ).compile(dtype=torch.float64)
    encoded = compiled.encode(
        [
            {"enabled": False, "child": 0},
            {"enabled": True, "child": 0},
            {"enabled": False, "child": 2},
        ]
    )

    keys = _semantics(compiled).key_tensor(encoded)

    assert keys.dtype == torch.int64
    assert keys.device == encoded.device
    assert torch.equal(keys[0], keys[2])
    assert not torch.equal(keys[0], keys[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_semantic_keys_cover_every_storage_transform_and_inactive_values(
    dtype: torch.dtype,
) -> None:
    compiled = Space(
        Bool("enabled"),
        Float("real", -1.0, 1.0, active_when=Eq("enabled", True)),
        Integer("linear", 0, 4, active_when=Eq("enabled", True)),
        Integer("stepped", 2, 10, step=2, active_when=Eq("enabled", True)),
        Integer("power", 1, 100, log=True, active_when=Eq("enabled", True)),
        Integer("exponent", 1, 16, exponent=True, base=2, active_when=Eq("enabled", True)),
        Categorical("category", ("a", "b"), active_when=Eq("enabled", True)),
        Bool("flag", active_when=Eq("enabled", True)),
        Integer("fixed", 3, 3, active_when=Eq("enabled", True)),
    ).compile(dtype=dtype)
    active_record = {
        "enabled": True,
        "real": 0.0,
        "linear": 2,
        "stepped": 6,
        "power": 4,
        "exponent": 8,
        "category": "a",
        "flag": False,
        "fixed": 3,
    }
    active = compiled.encode([active_record])

    equivalent = EncodedBatch(active.continuous.clone(), active.categorical.clone())
    equivalent.continuous[0, 0] = -0.0
    equivalent.continuous[0, 3] = math.log(4.4) / math.log(100.0)
    first = EncodedBatch(active.continuous.clone(), active.categorical.clone())
    first.continuous[0, 3] = math.log(4.1) / math.log(100.0)
    assert torch.equal(
        _semantics(compiled).key_tensor(first),
        _semantics(compiled).key_tensor(equivalent),
    )

    for parameter, value in (
        ("real", 0.5),
        ("linear", 3),
        ("stepped", 8),
        ("power", 5),
        ("exponent", 16),
        ("category", "b"),
        ("flag", True),
    ):
        record = active_record | {parameter: value}
        changed = compiled.encode([record])
        assert not torch.equal(
            _semantics(compiled).key_tensor(active),
            _semantics(compiled).key_tensor(changed),
        )

    inactive_a = compiled.encode(
        [
            {
                "enabled": False,
                "real": -0.75,
                "linear": 0,
                "stepped": 2,
                "power": 1,
                "exponent": 1,
                "category": "a",
                "flag": False,
                "fixed": 3,
            }
        ]
    )
    inactive_b = compiled.encode(
        [
            {
                "enabled": False,
                "real": 0.75,
                "linear": 4,
                "stepped": 10,
                "power": 100,
                "exponent": 16,
                "category": "b",
                "flag": True,
                "fixed": 3,
            }
        ]
    )
    assert torch.equal(
        _semantics(compiled).key_tensor(inactive_a),
        _semantics(compiled).key_tensor(inactive_b),
    )
    assert not torch.equal(
        _semantics(compiled).key_tensor(inactive_a),
        _semantics(compiled).key_tensor(active),
    )


@pytest.mark.parametrize("compiled_dtype", [torch.float32, torch.float64])
def test_semantics_normalize_input_dtype_and_log_integer_parent_values(
    compiled_dtype: torch.dtype,
) -> None:
    compiled = Space(
        Integer("parent", 1, 100, log=True),
        Bool("equal_child", active_when=Eq("parent", 4)),
        Bool("ordered_child", active_when=GreaterThan("parent", 4.2)),
    ).compile(dtype=compiled_dtype)
    canonical = compiled.encode([{"parent": 4, "equal_child": True}])
    input_dtype = torch.float64 if compiled_dtype == torch.float32 else torch.float32
    equivalent = EncodedBatch(
        canonical.continuous.to(dtype=input_dtype),
        canonical.categorical.clone(),
    )
    equivalent.continuous[0, 0] = math.log(4.4) / math.log(100.0)

    semantics = _semantics(compiled)
    canonical_activity = semantics.activity(canonical)
    equivalent_activity = semantics.activity(equivalent)

    assert canonical_activity.parameter.tolist() == [[True, True, False]]
    assert torch.equal(equivalent_activity.parameter, canonical_activity.parameter)
    assert torch.equal(
        semantics.key_tensor(equivalent),
        semantics.key_tensor(canonical),
    )
    assert compiled.decode(equivalent).to_records() == [{"parent": 4, "equal_child": True}]


def test_conditional_decode_uses_compiled_precision_at_rounding_boundaries() -> None:
    compiled = Space(
        Integer("parent", 1, 100, log=True),
        Bool("child", active_when=Eq("parent", 4)),
    ).compile(dtype=torch.float32)
    coordinate = math.log(4.5) / math.log(100.0) + 1e-12
    incoming = EncodedBatch(
        torch.tensor([[coordinate]], dtype=torch.float64),
        torch.tensor([[1]], dtype=torch.int64),
    )

    candidates = compiled.decode(incoming)

    assert candidates.dtype == torch.float32
    assert candidates.to_records() == [{"parent": 4, "child": True}]
    assert compiled.canonical_keys(candidates) == compiled.canonical_keys(
        compiled.encode(candidates.to_records())
    )


def test_numeric_predicates_use_semantic_thresholds_and_parent_guards() -> None:
    compiled = Space(
        Bool("enabled"),
        Integer("level", 2, 10, step=2, active_when=Eq("enabled", True)),
        Bool("high", active_when=GreaterThan("level", 5)),
    ).compile()
    candidates = compiled.decode(
        compiled.encode(
            [
                {"enabled": False, "level": 10},
                {"enabled": True, "level": 4},
                {"enabled": True, "level": 6, "high": True},
            ]
        )
    )

    assert candidates.activity is not None
    assert candidates.activity.tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_contextual_enumeration_handles_fixed_inactive_floats() -> None:
    compiled = Space(
        Float("child", 0.0, 1.0, active_when=Eq("branch", "on")),
        Categorical("branch", ("off", "on")),
    ).compile(dtype=torch.float64)

    assert not compiled.context_is_finite()
    with pytest.raises(ValueError, match="infinite"):
        list(compiled.iter_contextual_records())

    assert compiled.context_is_finite({"branch": "off"})
    disabled = list(compiled.iter_contextual_records({"branch": "off"}))
    assert disabled == [{"branch": "off"}]

    fixed_child = compiled.compile_fixed({"child": 0.375})
    candidates = list(compiled.iter_contextual_records(fixed_child))
    assert candidates == [
        {"branch": "off"},
        {"child": 0.375, "branch": "on"},
    ]


def test_context_uses_compiled_precision_for_fixed_float_predicates() -> None:
    rounded_inactive = Space(
        Float("parent", 0.0, 1.0),
        Float("child", 0.0, 1.0, active_when=GreaterThan("parent", 0.5)),
    ).compile(dtype=torch.float32)
    fixed = {"parent": 0.50000001}

    assert rounded_inactive.context_is_finite(fixed)
    candidates = list(rounded_inactive.iter_contextual_records(fixed))
    assert candidates == [{"parent": 0.50000001}]

    rounded_active = Space(
        Float("parent", 0.0, 1.0),
        Float("child", 0.0, 1.0, active_when=LessEqual("parent", 0.5)),
    ).compile(dtype=torch.float32)
    assert not rounded_active.context_is_finite(fixed)
    with pytest.raises(ValueError, match="infinite"):
        list(rounded_active.iter_contextual_records(fixed))


def test_finiteness_ignores_parameters_outside_float_condition_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = Space(
        Bool("gate"),
        *(Bool(f"unrelated_{index}") for index in range(25)),
        Float("child", 0.0, 1.0, active_when=Eq("gate", True)),
    ).compile()
    calls: list[str] = []
    original = conditional_module._finite_values

    def tracked(parameter: ParameterLike) -> tuple[object, ...]:
        calls.append(parameter.name)
        return original(parameter)

    monkeypatch.setattr(conditional_module, "_finite_values", tracked)

    assert compiled.context_is_finite({"gate": False})
    assert calls == []
    assert not compiled.context_is_finite()
    assert calls == ["gate"]


def test_lazy_enumeration_does_not_precount_a_large_finite_domain() -> None:
    selector = Bool("selector")
    children = tuple(
        Bool(f"child_{index}", active_when=Eq("selector", True)) for index in range(25)
    )
    compiled = Space(selector, *children).compile()

    assert compiled.context_is_finite()
    first = next(compiled.iter_contextual_records())

    assert first == {"selector": False}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_activity_projection_and_keys_have_cuda_parity() -> None:
    compiled = xgboost_space().compile(dtype=torch.float64)
    encoded = compiled.encode(
        [
            {"booster": "gblinear", "rate_drop": 0.8, "max_depth": 4},
            {"booster": "dart", "rate_drop": 0.3, "max_depth": 2},
        ]
    )
    cuda = encoded.to("cuda")

    cpu_candidates = compiled.decode(encoded)
    cuda_candidates = compiled.decode(cuda)

    assert cpu_candidates.activity is not None
    assert cuda_candidates.activity is not None
    assert torch.equal(cpu_candidates.activity, cuda_candidates.activity.cpu())
    assert cpu_candidates.to_records() == cuda_candidates.to_records()
    assert compiled.canonical_keys(encoded) == compiled.canonical_keys(cuda)


def test_conditional_decode_keeps_raw_tensor_identity() -> None:
    compiled = xgboost_space().compile()
    encoded = EncodedBatch(
        torch.tensor([[0.75, 4.0]], dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int64),
    )

    candidates = compiled.decode(encoded)

    assert candidates.continuous.data_ptr() == encoded.continuous.data_ptr()
    assert candidates.categorical.data_ptr() == encoded.categorical.data_ptr()
    assert candidates.to_records() == [{"booster": "gblinear"}]
