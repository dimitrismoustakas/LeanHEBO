# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from leanhebo.data import ObservationStore
from leanhebo.space import Bool, Integer, Space


@settings(max_examples=30, deadline=None)
@given(values=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=30))
def test_duplicate_membership_never_removes_a_distinct_point(values: list[int]) -> None:
    compiled = Space(Integer("x", 0, 10), Bool("flag")).compile()
    records = [{"x": value, "flag": bool(value % 2)} for value in values]
    candidates = compiled.decode(compiled.encode(records))
    store = ObservationStore(compiled)
    mask = store.unique_mask(candidates)
    selected_keys = compiled.canonical_keys(candidates.encoded.select(mask))
    assert len(selected_keys) == len(set(selected_keys))
    assert set(selected_keys) == set(compiled.canonical_keys(candidates))


@settings(max_examples=25, deadline=None)
@given(chunk_sizes=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=10))
def test_chunked_materialization_preserves_append_order(chunk_sizes: list[int]) -> None:
    compiled = Space(Integer("x", 0, 100)).compile()
    store = ObservationStore(compiled)
    expected: list[float] = []
    offset = 0
    for chunk_size in chunk_sizes:
        records = [{"x": (offset + index) % 101} for index in range(chunk_size)]
        outcomes = [float(offset + index) for index in range(chunk_size)]
        store.append(records, outcomes)
        expected.extend(outcomes)
        offset += chunk_size
    assert store.chunk_count == len(chunk_sizes)
    assert torch.equal(store.y, torch.tensor(expected))
