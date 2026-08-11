# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from leanhebo.data.store import ObservationStore
from leanhebo.space import Bool, Categorical, Float, Integer, Space


def test_public_snapshots_cannot_mutate_append_only_history() -> None:
    compiled = Space(Integer("x", 0, 5), Bool("b")).compile()
    candidates = compiled.sample(3, seed=11)
    store = ObservationStore(compiled)
    store.append(candidates, [1.0, 2.0, 3.0])
    expected = store.materialize()

    materialized = store.materialize()
    materialized.continuous.zero_()
    materialized.categorical.zero_()
    materialized.y.fill_(99)

    actual = store.materialize()
    assert torch.equal(actual.continuous, expected.continuous)
    assert torch.equal(actual.categorical, expected.categorical)
    assert torch.equal(actual.y, expected.y)
    assert not store.unique_mask(candidates).any()


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


def test_observed_rows_are_excluded_from_future_unique_batches() -> None:
    compiled = Space(Integer("x", 0, 5), Categorical("c", ("a", "b"))).compile()
    records = [
        {"x": 1, "c": "a"},
        {"x": 1, "c": "a"},
        {"x": 2, "c": "a"},
    ]
    candidates = compiled.decode(compiled.encode(records))
    store = ObservationStore(compiled)
    assert store.unique_mask(candidates).tolist() == [True, False, True]
    store.append(candidates, [1.0, 2.0, 3.0])
    assert store.unique_mask(candidates).tolist() == [False, False, False]


def test_float_canonical_keys_normalize_signed_zero() -> None:
    compiled = Space(Float("x", -1.0, 1.0)).compile()
    negative_zero = compiled.decode(compiled.encode([{"x": -0.0}]))
    positive_zero = compiled.decode(compiled.encode([{"x": 0.0}]))
    store = ObservationStore(compiled)

    assert compiled.canonical_keys(negative_zero) == compiled.canonical_keys(positive_zero)
    store.append(negative_zero, [1.0])
    assert store.unique_mask(positive_zero).tolist() == [False]


def test_transformed_values_are_invalidated_when_observations_change() -> None:
    compiled = Space(Float("x", 0.0, 1.0)).compile(dtype=torch.float64)
    store = ObservationStore(compiled)
    store.append(compiled.sample(2, seed=7), [1.0, 2.0])
    store.set_transformed_y([-1.0, 1.0])

    assert torch.equal(store.transformed_y, torch.tensor([-1.0, 1.0], dtype=torch.float64))
    store.append(compiled.sample(1, seed=8), [3.0])
    assert store.transformed_y is None
