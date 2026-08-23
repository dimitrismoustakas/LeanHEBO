# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from leanhebo import LeanHEBO
from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig, WarpConfig
from leanhebo.errors import SearchSpaceExhaustedError
from leanhebo.gp import ConditionalExactGPSurrogate
from leanhebo.space import Categorical, Eq, Float, In, Integer, Space


def _config(*, random_samples: int, seed: int = 31) -> LeanHEBOConfig:
    return LeanHEBOConfig(
        random_samples=random_samples,
        runtime=RuntimeConfig(seed=seed, acquisition_batch_size=None),
        gp=GPConfig(
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
        ),
        warp=WarpConfig(method="none"),
        search=SearchConfig(
            population_size=8,
            generations=1,
            seed=seed + 1,
        ),
    )


def _xgboost_space() -> Space:
    return Space(
        Categorical("booster", ("gblinear", "gbtree", "dart")),
        Float("reg_lambda", 1e-4, 1.0, log=True),
        Integer("max_depth", 2, 8, active_when=In("booster", ("gbtree", "dart"))),
        Float("rate_drop", 0.0, 1.0, active_when=Eq("booster", "dart")),
    )


def _objective(records: list[dict[str, object]]) -> torch.Tensor:
    values: list[float] = []
    branch_cost = {"gblinear": 0.8, "gbtree": 0.3, "dart": 0.1}
    for record in records:
        value = branch_cost[str(record["booster"])] + float(record["reg_lambda"])
        if "max_depth" in record:
            value += (int(record["max_depth"]) - 5) ** 2 / 25
        if "rate_drop" in record:
            value += (float(record["rate_drop"]) - 0.2) ** 2
        values.append(value)
    return torch.tensor(values)


def test_xgboost_space_runs_one_global_conditional_gp_and_semantic_search() -> None:
    optimizer = LeanHEBO(_xgboost_space(), config=_config(random_samples=5))
    initial = optimizer.suggest(5)

    for record in initial.to_records():
        booster = record["booster"]
        assert ("max_depth" in record) is (booster in {"gbtree", "dart"})
        assert ("rate_drop" in record) is (booster == "dart")
    optimizer.observe(initial, _objective(initial.to_records()))

    modeled = optimizer.suggest(3)

    assert isinstance(optimizer.surrogate, ConditionalExactGPSurrogate)
    assert optimizer.surrogate.train_activity is not None
    assert optimizer.surrogate.train_activity.shape == (5, 2)
    assert optimizer.store.unique_mask(modeled).all()
    assert optimizer.last_search is not None
    population = optimizer.space.candidate_from_dense(optimizer.last_search.population)
    keys = optimizer.space.canonical_keys(population)
    assert len(keys) == len(set(keys))


def test_fixed_child_makes_the_context_finite_without_forcing_its_parent() -> None:
    optimizer = LeanHEBO(
        Space(
            Categorical("branch", ("off", "on")),
            Float("child", 0.0, 1.0, active_when=Eq("branch", "on")),
        ),
        config=_config(random_samples=64),
    )

    candidates = optimizer.suggest(2, fix_input={"child": 0.375})

    assert candidates.to_records() == [
        {"branch": "off"},
        {"branch": "on", "child": 0.375},
    ]
    optimizer.observe(candidates, torch.tensor([0.0, 1.0]))
    with pytest.raises(SearchSpaceExhaustedError, match=r"only 0 unseen"):
        optimizer.suggest(1, fix_input={"child": 0.375})


def test_contextual_incumbent_ignores_a_fixed_child_while_it_is_inactive() -> None:
    optimizer = LeanHEBO(
        Space(
            Categorical("branch", ("off", "on")),
            Float("root", 0.0, 1.0),
            Float("child", 0.0, 1.0, active_when=Eq("branch", "on")),
        ),
        config=_config(random_samples=2),
    )
    observations = optimizer.space.decode(
        optimizer.space.encode(
            [
                {"branch": "off", "root": 0.1, "child": 0.2},
                {"branch": "on", "root": 0.9, "child": 0.4},
            ]
        )
    )
    optimizer.observe(observations, torch.tensor([0.0, 1.0]))

    incumbent = optimizer._incumbent(optimizer.space.compile_fixed({"child": 0.8}))

    assert incumbent.to_records() == [{"branch": "off", "root": pytest.approx(0.1)}]


def test_conditional_checkpoint_continuation_is_exact(tmp_path: Path) -> None:
    optimizer = LeanHEBO(_xgboost_space(), config=_config(random_samples=4, seed=73))
    initial = optimizer.suggest(4)
    optimizer.observe(initial, _objective(initial.to_records()))
    optimizer.suggest(1)
    path = tmp_path / "conditional.leanhebo"
    optimizer.save(path)

    restored = LeanHEBO.load(path)
    expected = optimizer.suggest(2)
    actual = restored.suggest(2)

    assert isinstance(restored.surrogate, ConditionalExactGPSurrogate)
    assert torch.equal(actual.continuous, expected.continuous)
    assert torch.equal(actual.categorical, expected.categorical)
    assert actual.to_records() == expected.to_records()
