# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from leanhebo.data import CandidateBatch, EncodedBatch
from leanhebo.space import Bool, Categorical, Float, Integer, Space


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
    assert Integer("x", 1, 100, log=True).mode == "power"
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_integer_codec_has_exact_endpoints_and_idempotent_repair(
    dtype: torch.dtype,
) -> None:
    compiled = Space(Integer("batch_size", 16, 512, log=True)).compile(dtype=dtype)
    encoded = compiled.encode([{"batch_size": 16}, {"batch_size": 512}])

    assert torch.equal(encoded.continuous[:, 0], torch.tensor([0.0, 1.0], dtype=dtype))
    compiled.validate_encoded(encoded)

    raw = torch.tensor([[-1.0], [0.37], [2.0]], dtype=dtype)
    repaired = compiled.encoded_from_dense(raw)
    repaired_twice = compiled.encoded_from_dense(compiled.to_dense(repaired))
    assert torch.equal(repaired.continuous, repaired_twice.continuous)
    assert compiled.decode(repaired).to_records()[0]["batch_size"] == 16
    assert compiled.decode(repaired).to_records()[-1]["batch_size"] == 512


def test_float32_log_integer_encoding_preserves_semantic_round_trip() -> None:
    compiled = Space(Integer("n", 8476, 825012, log=True)).compile(dtype=torch.float32)
    encoded = compiled.encode([{"n": 617012}])

    assert compiled.decode(encoded).to_records() == [{"n": 617012}]
    repaired = compiled.encoded_from_dense(encoded.continuous)
    repaired_twice = compiled.encoded_from_dense(repaired.continuous)
    assert torch.equal(repaired.continuous, repaired_twice.continuous)


def test_static_dimensions_are_absent_from_tensors_and_reinserted_on_decode() -> None:
    compiled = Space(
        Categorical("dataset", ("protein",)),
        Integer("fold", 3, 3),
        Float("learning_rate", 1e-4, 1e-1, log=True),
    ).compile()
    record = {"dataset": "protein", "fold": 3, "learning_rate": 1e-2}

    encoded = compiled.encode([record])

    assert compiled.dense_dimension == 1
    assert compiled.n_continuous == 1
    assert compiled.n_categorical == 0
    assert encoded.continuous.shape == (1, 1)
    assert encoded.categorical.shape == (1, 0)
    assert compiled.decode(encoded).to_records() == [record]


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


def test_numpy_and_column_mapping_adapters() -> None:
    compiled = Space(Float("x", -1.0, 1.0), Integer("n", 1, 3), Bool("flag")).compile()
    array = np.asarray([[0.25, 2, True], [-0.5, 1, False]], dtype=object)
    candidates = compiled.decode(compiled.encode(array))
    mapping = {"x": [0.25, -0.5], "n": [2, 1], "flag": [True, False]}
    tensor_mapping = {
        "x": torch.tensor([0.25, -0.5]),
        "n": torch.tensor([2, 1]),
        "flag": torch.tensor([True, False]),
    }

    assert candidates.to_records() == [
        {"x": 0.25, "n": 2, "flag": True},
        {"x": -0.5, "n": 1, "flag": False},
    ]
    assert compiled.decode(compiled.encode(mapping)).to_records() == candidates.to_records()
    assert compiled.decode(compiled.encode(tensor_mapping)).to_records() == candidates.to_records()


def test_dense_bridge_repairs_every_variable_kind() -> None:
    compiled = mixed_space().compile()
    raw = torch.tensor(
        [[-20.0, 3.4, 1.6, math.log(4.3) / math.log(100), 7.6, 9.0, -2.0]],
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
    record = {"category": "b", "x": 0.25, "flag": True, "n": 2}
    assert compiled.decode(compiled.encode([record])).to_records() == [record]


def test_sobol_sampling_is_deterministic_and_fixed_values_are_exact() -> None:
    compiled = mixed_space().compile(dtype=torch.float64)
    fixed = compiled.compile_fixed(
        {"learning_rate": 0.003, "depth": 11, "activation": "silu", "use_bias": True}
    )
    first = compiled.sample(32, seed=17, fixed=fixed)
    second = compiled.sample(32, seed=17, fixed=fixed)

    assert torch.equal(first.continuous, second.continuous)
    assert torch.equal(first.categorical, second.categorical)
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
        torch.tensor([[math.log(4.1) / math.log(100), 0.25]], dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.int64),
    )
    second = EncodedBatch(
        torch.tensor([[math.log(4.4) / math.log(100), 0.25]], dtype=torch.float64),
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
