# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from leanhebo.data import CandidateBatch, EncodedBatch
from leanhebo.space import Bool, Categorical, DenseKind, Float, Integer, Space


def mixed_space() -> Space:
    return Space(
        Float("learning_rate", 1e-5, 1e-1, log=True),
        Integer("depth", 1, 20),
        Integer("width", 4, 16, step=4),
        Integer("workers", 1, 100, log=True),
        Integer("batch_size", 32, 1024, base=2, exponent=True),
        Categorical("activation", ("relu", "gelu", "silu")),
        Bool("use_bias"),
    )


def test_parameter_validation_and_transform_modes() -> None:
    assert Integer("x", 4, 16, step=4).optimization_bounds == (0.0, 3.0)
    assert Integer("x", 1, 100, power=True).mode == "power"
    assert Integer("x", 32, 1024, base=2, exponent=True).optimization_bounds == (
        5.0,
        10.0,
    )
    with pytest.raises(ValueError, match="divisible"):
        Integer("x", 0, 10, step=4)
    with pytest.raises(ValueError, match="exact integral powers"):
        Integer("x", 30, 1024, base=2, exponent=True)
    with pytest.raises(ValueError, match="duplicate category"):
        Categorical("choice", (True, 1))


def test_log_boundaries_and_numpy_scalars_encode_cleanly() -> None:
    compiled = Space(
        Float("x", 1e-9, 1e9, log=True, base=2),
        Integer("n", 1, 1000, log=True, base=2),
        Categorical("category", (np.int64(1), np.int64(2))),
        Bool("flag"),
    ).compile()
    encoded = compiled.encode(
        [
            {"x": 1e-9, "n": 1, "category": np.int64(1), "flag": np.bool_(True)},
            {"x": 1e9, "n": 1000, "category": np.int64(2), "flag": np.bool_(False)},
        ]
    )
    compiled.validate_encoded(encoded)
    assert compiled.decode(encoded).to_records()[0]["category"] == 1


def test_encode_decode_round_trip_and_tensor_identity() -> None:
    compiled = mixed_space().compile(dtype=torch.float64)
    records = [
        {
            "learning_rate": 1e-3,
            "depth": 7,
            "width": 12,
            "workers": 27,
            "batch_size": 256,
            "activation": "gelu",
            "use_bias": True,
        },
        {
            "learning_rate": 2e-4,
            "depth": 1,
            "width": 4,
            "workers": 3,
            "batch_size": 32,
            "activation": "relu",
            "use_bias": False,
        },
    ]
    encoded = compiled.encode(records)
    candidates = compiled.decode(encoded)

    assert candidates.continuous.data_ptr() == encoded.continuous.data_ptr()
    assert candidates.categorical.data_ptr() == encoded.categorical.data_ptr()
    decoded_records = candidates.to_records()
    assert decoded_records[0]["learning_rate"] == pytest.approx(1e-3)
    assert decoded_records[1]["learning_rate"] == pytest.approx(2e-4)
    for decoded, expected in zip(decoded_records, records, strict=True):
        assert {key: value for key, value in decoded.items() if key != "learning_rate"} == {
            key: value for key, value in expected.items() if key != "learning_rate"
        }
    assert compiled.encode(candidates).continuous.data_ptr() == encoded.continuous.data_ptr()
    assert candidates.to_numpy().shape == (2, len(compiled))


def test_numpy_pandas_and_polars_adapters() -> None:
    compiled = Space(Float("x", -1.0, 1.0), Integer("n", 1, 3), Bool("flag")).compile()
    array = np.asarray([[0.25, 2, True], [-0.5, 1, False]], dtype=object)
    candidates = compiled.decode(compiled.encode(array))

    pandas_frame = candidates.to_pandas()
    polars_frame = candidates.to_polars()
    assert compiled.decode(compiled.encode(pandas_frame)).to_records() == candidates.to_records()
    assert compiled.decode(compiled.encode(polars_frame)).to_records() == candidates.to_records()


def test_compiled_metadata_and_dense_bridge() -> None:
    compiled = mixed_space().compile()
    assert compiled.dense_names == (
        "learning_rate",
        "depth",
        "width",
        "workers",
        "batch_size",
        "activation",
        "use_bias",
    )
    assert compiled.dense_kind_codes.tolist() == [
        DenseKind.FLOAT,
        DenseKind.INTEGER,
        DenseKind.STEPPED_INTEGER,
        DenseKind.POWER_INTEGER,
        DenseKind.EXPONENT_INTEGER,
        DenseKind.CATEGORICAL,
        DenseKind.BOOLEAN,
    ]
    assert compiled.real_mask.tolist() == [True, False, False, False, False, False, False]
    assert compiled.integer_mask.tolist() == [False, True, True, True, True, False, False]
    assert compiled.categorical_mask.tolist() == [False, False, False, False, False, True, True]
    assert compiled.search_integer_mask.tolist() == [
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert compiled.search_continuous_mask.tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]

    raw = torch.tensor(
        [[-20.0, 3.4, 1.6, math.log10(4.3), 7.6, 9.0, -2.0]],
        dtype=torch.float32,
    )
    encoded = compiled.encoded_from_dense(raw)
    repaired = compiled.to_dense(encoded)
    assert torch.all(repaired >= compiled.dense_lower_bounds)
    assert torch.all(repaired <= compiled.dense_upper_bounds)
    assert repaired[0, 1:3].tolist() == [3.0, 2.0]
    assert repaired[0, 4:].tolist() == [8.0, 2.0, 0.0]
    assert compiled.decode(encoded).to_records()[0]["workers"] == 4


def test_public_column_order_is_independent_of_dense_tensor_order() -> None:
    compiled = Space(
        Categorical("category", ("a", "b")),
        Float("x", 0.0, 1.0),
        Bool("flag"),
        Integer("n", 1, 3),
    ).compile()
    assert compiled.names == ("category", "x", "flag", "n")
    assert compiled.dense_names == ("x", "n", "category", "flag")
    assert compiled.public_to_dense_indices.tolist() == [2, 0, 3, 1]
    assert compiled.dense_to_public_indices.tolist() == [1, 3, 0, 2]
    record = {"category": "b", "x": 0.25, "flag": True, "n": 2}
    assert compiled.decode(compiled.encode([record])).to_records() == [record]
    assert compiled.category_to_code["category"]["b"] == 1
    assert compiled.code_to_category["flag"] == (False, True)


def test_sobol_sampling_is_deterministic_and_fixed_values_are_exact() -> None:
    compiled = mixed_space().compile(dtype=torch.float64)
    fixed = compiled.compile_fixed(
        {"learning_rate": 0.003, "depth": 11, "activation": "silu", "use_bias": True}
    )
    first = compiled.sample(32, seed=17, fixed=fixed)
    second = compiled.sample(32, seed=17, fixed=fixed)

    assert torch.equal(first.continuous, second.continuous)
    assert torch.equal(first.categorical, second.categorical)
    assert compiled.fixed_mask(fixed).tolist() == [True, True, False, False, False, True, True]
    dense_fixed = compiled.dense_fixed_values(fixed)
    assert dense_fixed[compiled.fixed_mask(fixed)].tolist() == pytest.approx(
        [-2.5228787452803374, 11.0, 2.0, 1.0]
    )
    for record in first.to_records():
        assert record["learning_rate"] == 0.003
        assert record["depth"] == 11
        assert record["activation"] == "silu"
        assert record["use_bias"] is True


def test_schema_spec_round_trip_has_stable_fingerprint() -> None:
    space = mixed_space()
    restored = Space.from_spec(space.to_spec())
    assert restored == space
    assert restored.compile().fingerprint == space.compile().fingerprint


def test_canonical_keys_follow_semantic_integer_values() -> None:
    compiled = Space(Integer("n", 1, 100, log=True), Float("x", -1.0, 1.0)).compile(
        dtype=torch.float64
    )
    # Distinct search coordinates that both decode to integer 4 are one point.
    first = EncodedBatch(
        torch.tensor([[math.log10(4.1), 0.25]], dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.int64),
    )
    second = EncodedBatch(
        torch.tensor([[math.log10(4.4), 0.25]], dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.int64),
    )
    assert compiled.canonical_keys(first) == compiled.canonical_keys(second)


def test_candidate_device_and_dtype_transfer_keeps_decoded_values() -> None:
    compiled = Space(Float("x", 0.0, 1.0), Categorical("c", ("a", "b"))).compile()
    original = compiled.sample(3, seed=2)
    moved = original.to(dtype=torch.float64)
    assert isinstance(moved, CandidateBatch)
    assert moved.dtype == torch.float64
    assert moved.categorical.dtype == torch.int64
    assert moved.to_records() == original.to_records()
