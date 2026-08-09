# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from leanhebo import LeanHEBO
from leanhebo.config import LeanHEBOConfig, RuntimeConfig
from leanhebo.data import CandidateBatch
from leanhebo.space import Bool, Categorical, Float, Integer, Space


def _mixed_space() -> Space:
    return Space(
        Float("rate", 1e-5, 1e-1, log=True),
        Integer("width", 4, 24, step=4),
        Integer("workers", 1, 100, log=True),
        Integer("batch", 8, 256, base=2, exponent=True),
        Categorical("activation", ("relu", "gelu", "silu")),
        Bool("bias"),
    )


def _assert_valid_context(
    candidates: CandidateBatch,
    *,
    rate: float,
    activation: str,
    bias: bool,
) -> None:
    for record in candidates.to_records():
        assert record["rate"] == rate
        assert record["activation"] == activation
        assert record["bias"] is bias
        assert record["width"] in range(4, 25, 4)
        assert 1 <= int(record["workers"]) <= 100
        assert record["batch"] in (8, 16, 32, 64, 128, 256)


@settings(max_examples=24, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**16),
    first_count=st.integers(min_value=1, max_value=12),
    second_count=st.integers(min_value=1, max_value=12),
    rate=st.sampled_from((1e-5, 3e-4, 1e-2, 1e-1)),
    activation=st.sampled_from(("relu", "gelu", "silu")),
    bias=st.booleans(),
    precompile_fixed=st.booleans(),
)
def test_optimizer_emits_valid_unique_mixed_candidates_with_exact_context(
    seed: int,
    first_count: int,
    second_count: int,
    rate: float,
    activation: str,
    bias: bool,
    precompile_fixed: bool,
) -> None:
    optimizer = LeanHEBO(
        _mixed_space(),
        config=LeanHEBOConfig(
            random_samples=64,
            runtime=RuntimeConfig(seed=seed, dtype="float64"),
        ),
    )
    assignments = {"rate": rate, "activation": activation, "bias": bias}
    fixed = optimizer.space.compile_fixed(assignments) if precompile_fixed else assignments

    first = optimizer.suggest(first_count, fix_input=fixed)
    optimizer.space.validate_encoded(first.encoded)
    first_keys = optimizer.space.canonical_keys(first)
    assert len(first_keys) == len(set(first_keys)) == first_count
    _assert_valid_context(first, rate=rate, activation=activation, bias=bias)

    optimizer.observe(first, torch.arange(first_count, dtype=torch.float64))
    second = optimizer.suggest(second_count, fix_input=fixed)
    optimizer.space.validate_encoded(second.encoded)
    second_keys = optimizer.space.canonical_keys(second)
    assert len(second_keys) == len(set(second_keys)) == second_count
    assert set(first_keys).isdisjoint(second_keys)
    _assert_valid_context(second, rate=rate, activation=activation, bias=bias)
