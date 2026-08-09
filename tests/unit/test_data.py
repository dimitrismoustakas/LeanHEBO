# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from leanhebo.data import EncodedBatch, ObservationStore
from leanhebo.space import Bool, Categorical, CompiledSpace, Float, Integer, Space


def test_observe_candidate_uses_direct_encoded_path(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = Space(Float("x", 0.0, 1.0), Integer("n", 1, 5)).compile()
    candidates = compiled.sample(4, seed=1)
    store = ObservationStore(compiled)

    def fail_encode(self: CompiledSpace, value: object) -> EncodedBatch:
        raise AssertionError(f"unexpected re-encoding of {type(value).__name__}")

    monkeypatch.setattr(CompiledSpace, "encode", fail_encode)
    assert store.observe(candidates, torch.arange(4.0)) == 4
    assert store.encoded_chunks[0].continuous.data_ptr() != candidates.continuous.data_ptr()
    assert store.contains(candidates).all()


def test_chunks_materialize_once_and_invalidate_on_append() -> None:
    compiled = Space(Float("x", -1.0, 1.0), Bool("b")).compile()
    store = ObservationStore(compiled)
    store.append(compiled.sample(2, seed=1), [1.0, 2.0])
    store.append(compiled.sample(3, seed=2), [3.0, 4.0, 5.0])

    assert store.chunk_count == 2
    first = store.materialize()
    second = store.materialize()
    assert first is not second
    assert torch.equal(first.continuous, second.continuous)
    assert first.continuous.data_ptr() != second.continuous.data_ptr()
    assert len(first) == 5
    store.append(compiled.sample(1, seed=3), [6.0])
    assert len(store.materialize()) == 6
    assert len(store) == 6


def test_public_snapshots_cannot_mutate_append_only_history() -> None:
    compiled = Space(Integer("x", 0, 5), Bool("b")).compile()
    candidates = compiled.sample(3, seed=11)
    store = ObservationStore(compiled)
    store.append(candidates, [1.0, 2.0, 3.0])
    expected = store.materialize()

    chunks = store.encoded_chunks
    outcomes = store.y_chunks
    materialized = store.materialize()
    chunks[0].categorical.zero_()
    outcomes[0].fill_(99)
    materialized.continuous.zero_()
    materialized.y.fill_(99)

    actual = store.materialize()
    assert torch.equal(actual.continuous, expected.continuous)
    assert torch.equal(actual.categorical, expected.categorical)
    assert torch.equal(actual.y, expected.y)
    assert store.contains(candidates).all()


def test_nonfinite_policy_and_output_shape_validation() -> None:
    compiled = Space(Integer("x", 0, 5)).compile()
    candidates = compiled.sample(3, seed=4)
    dropping = ObservationStore(compiled, nonfinite="drop")
    assert dropping.append(candidates, [1.0, float("nan"), 3.0]) == 2
    assert dropping.discarded_count == 1
    assert torch.equal(dropping.y, torch.tensor([1.0, 3.0]))

    strict = ObservationStore(compiled, nonfinite="raise")
    with pytest.raises(ValueError, match="non-finite"):
        strict.append(candidates, [1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match="batch lengths"):
        strict.append(candidates, [1.0])
    with pytest.raises(ValueError, match="single-objective"):
        strict.append(candidates, torch.ones(3, 2))


def test_duplicate_membership_and_pending_rows() -> None:
    compiled = Space(Integer("x", 0, 5), Categorical("c", ("a", "b"))).compile()
    records = [
        {"x": 1, "c": "a"},
        {"x": 1, "c": "a"},
        {"x": 2, "c": "a"},
    ]
    candidates = compiled.decode(compiled.encode(records))
    store = ObservationStore(compiled)
    assert store.unique_mask(candidates).tolist() == [True, False, True]
    assert store.add_keys(candidates) == 2
    assert store.contains(candidates).tolist() == [True, True, True]


def test_transformed_values_are_versioned_and_invalidated() -> None:
    compiled = Space(Float("x", 0.0, 1.0)).compile(dtype=torch.float64)
    store = ObservationStore(compiled)
    store.append(compiled.sample(2, seed=7), [1.0, 2.0])
    raw_version = store.observation_version
    store.set_transformed_y([-1.0, 1.0], observation_version=raw_version)

    assert not store.transform_is_stale
    assert store.transform_version == 1
    assert torch.equal(store.transformed_y, torch.tensor([-1.0, 1.0], dtype=torch.float64))
    store.append(compiled.sample(1, seed=8), [3.0])
    assert store.transform_is_stale
    assert store.transformed_y is None


def test_decoded_record_retention_is_opt_in() -> None:
    compiled = Space(Float("x", 0.0, 1.0), Bool("b")).compile()
    candidates = compiled.sample(3, seed=4, fixed={"x": 0.125})
    store = ObservationStore(compiled, retain_decoded=True)
    store.append(candidates, [1.0, 2.0, 3.0])
    assert store.records == candidates.to_records()
