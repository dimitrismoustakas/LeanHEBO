# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import torch

from leanhebo import LeanHEBO
from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig
from leanhebo.space import Bool, Categorical, Float, Integer, Space


def _config(*, seed: int = 9, keep_history: bool = False) -> LeanHEBOConfig:
    return LeanHEBOConfig(
        random_samples=3,
        runtime=RuntimeConfig(seed=seed, acquisition_batch_size=32),
        gp=GPConfig(
            optimizer="adam",
            initial_steps=2,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
        ),
        search=SearchConfig(
            population_size=12,
            generations=2,
            seed=seed + 1,
            keep_history=keep_history,
        ),
    )


def _space() -> Space:
    return Space(
        Float("x", -2.0, 2.0),
        Integer("depth", 1, 5),
        Categorical("kind", ("a", "b", "c")),
        Bool("enabled"),
    )


def _objective(records: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [
            float(row["x"]) ** 2
            + (int(row["depth"]) - 3) ** 2
            + (0.0 if row["kind"] == "a" else 0.5)
            for row in records
        ]
    )


def test_mixed_sequential_loop_uses_direct_candidate_observation_and_warm_update() -> None:
    optimizer = LeanHEBO(_space(), config=_config())
    initial = optimizer.suggest(3)
    assert optimizer.observe(initial, _objective(initial.to_records())) == 3

    model_candidates = optimizer.suggest(2, fix_input={"enabled": True})
    assert all(row["enabled"] is True for row in model_candidates.to_records())
    assert optimizer.surrogate is not None
    model_identity = id(optimizer.surrogate.model)
    assert optimizer.observe(model_candidates, _objective(model_candidates.to_records())) == 2

    following = optimizer.suggest(2, fix_input={"enabled": True})
    assert len(following) == 2
    assert optimizer.surrogate is not None
    assert id(optimizer.surrogate.model) == model_identity
    assert optimizer.surrogate.update_count == 1
    assert optimizer.best_y == float(optimizer.store.y.min())
    assert optimizer.diagnostics.counters["posterior.calls"] > 0


def test_nonfinite_observations_are_dropped_by_documented_default() -> None:
    optimizer = LeanHEBO(_space(), config=_config())
    candidates = optimizer.suggest(3)
    retained = optimizer.observe(candidates, np.asarray([1.0, np.nan, 2.0]))
    assert retained == 2
    assert optimizer.observations == 2
    assert optimizer.store.discarded_count == 1
    assert optimizer.diagnostics.counters["observe.discarded"] == 1


def test_trained_checkpoint_restores_rng_model_optimizer_and_continuation(tmp_path: object) -> None:
    optimizer = LeanHEBO(_space(), config=_config(seed=21, keep_history=True))
    initial = optimizer.suggest(3)
    optimizer.observe(initial, _objective(initial.to_records()))
    optimizer.suggest(2)
    assert optimizer.surrogate is not None

    checkpoint = tmp_path / "run.leanhebo"  # type: ignore[operator]
    optimizer.save(checkpoint)
    restored = LeanHEBO.load(checkpoint, map_location="cpu")
    assert restored.surrogate is not None
    assert restored.observations == optimizer.observations
    assert restored.store.observation_version == optimizer.store.observation_version
    assert restored.store.transform_version == optimizer.store.transform_version
    assert restored.store.transform_is_stale == optimizer.store.transform_is_stale
    assert restored.surrogate.fit_count == optimizer.surrogate.fit_count
    assert len(restored.search_history) == len(optimizer.search_history) == 1
    for actual_history, expected_history in zip(
        restored.search_history, optimizer.search_history, strict=True
    ):
        torch.testing.assert_close(actual_history[0], expected_history[0])
        torch.testing.assert_close(actual_history[1], expected_history[1])

    expected = optimizer.suggest(3, fix_input={"kind": "b"})
    actual = restored.suggest(3, fix_input={"kind": "b"})
    torch.testing.assert_close(actual.continuous, expected.continuous)
    assert torch.equal(actual.categorical, expected.categorical)
    assert actual.to_records() == expected.to_records()


def test_checkpoint_restores_reserved_keys_discard_counts_and_diagnostics(tmp_path: object) -> None:
    optimizer = LeanHEBO(_space(), config=_config(seed=33))
    observed = optimizer.space.decode(
        optimizer.space.encode(
            [
                {"x": -1.5, "depth": 1, "kind": "a", "enabled": False},
                {"x": -1.0, "depth": 2, "kind": "b", "enabled": True},
            ]
        )
    )
    optimizer.observe(observed, [1.0, float("nan")])
    reserved = optimizer.space.decode(
        optimizer.space.encode([{"x": 1.5, "depth": 5, "kind": "c", "enabled": True}])
    )
    optimizer.store.add_keys(reserved)
    observation_version = optimizer.store.observation_version
    transform_version = optimizer.store.transform_version
    optimizer.diagnostics.increment("custom.audit.counter", 7)

    checkpoint = tmp_path / "auxiliary-state.leanhebo"  # type: ignore[operator]
    optimizer.save(checkpoint)
    restored = LeanHEBO.load(checkpoint, map_location="cpu")

    assert restored.store.contains(reserved).all()
    assert restored.store.observation_version == observation_version
    assert restored.store.transform_version == transform_version
    assert restored.store.discarded_count == 1
    assert restored.diagnostics.counters["observe.discarded"] == 1
    assert restored.diagnostics.counters["custom.audit.counter"] == 7
