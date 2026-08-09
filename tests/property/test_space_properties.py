# SPDX-License-Identifier: MIT

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from leanhebo.space import Bool, Categorical, Float, Integer, Space


@settings(max_examples=40, deadline=None)
@given(
    x=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    integer=st.integers(min_value=-5, max_value=5),
    step_index=st.integers(min_value=0, max_value=5),
    category=st.sampled_from(("red", "green", "blue")),
    flag=st.booleans(),
)
def test_external_encode_decode_cycle(
    x: float, integer: int, step_index: int, category: str, flag: bool
) -> None:
    compiled = Space(
        Float("x", -10.0, 10.0),
        Integer("integer", -5, 5),
        Integer("stepped", 4, 24, step=4),
        Categorical("category", ("red", "green", "blue")),
        Bool("flag"),
    ).compile(dtype="float64")
    record = {
        "x": x,
        "integer": integer,
        "stepped": 4 + step_index * 4,
        "category": category,
        "flag": flag,
    }
    decoded = compiled.decode(compiled.encode([record])).to_records()[0]
    assert decoded == record


@settings(max_examples=30, deadline=None)
@given(count=st.integers(min_value=0, max_value=64), seed=st.integers(min_value=0, max_value=9999))
def test_generated_candidates_are_valid_and_context_is_exact(count: int, seed: int) -> None:
    compiled = Space(
        Float("positive", 1e-4, 1e2, log=True),
        Integer("step", 10, 50, step=10),
        Integer("power", 1, 1000, log=True),
        Integer("exponent", 8, 256, base=2, exponent=True),
        Categorical("choice", ("a", "b", "c")),
        Bool("enabled"),
    ).compile(dtype="float64")
    candidates = compiled.sample(
        count,
        seed=seed,
        fixed={"positive": 0.003, "step": 30, "choice": "b", "enabled": False},
    )
    compiled.validate_encoded(candidates.encoded)
    for record in candidates.to_records():
        assert record["positive"] == 0.003
        assert record["step"] == 30
        assert record["choice"] == "b"
        assert record["enabled"] is False


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=9999), count=st.integers(min_value=1, max_value=32))
def test_dense_round_trip_preserves_canonical_candidates(seed: int, count: int) -> None:
    compiled = Space(
        Float("x", -2.0, 3.0),
        Integer("n", 1, 20, log=True),
        Integer("k", 1, 5),
        Categorical("c", (0, 1, 2)),
    ).compile()
    first = compiled.sample(count, seed=seed)
    second = compiled.candidate_from_dense(compiled.to_dense(first))
    assert compiled.canonical_keys(first) == compiled.canonical_keys(second)
    assert first.to_records() == second.to_records()
