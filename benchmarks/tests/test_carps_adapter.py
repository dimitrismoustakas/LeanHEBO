from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("carps")
pytest.importorskip("ConfigSpace")

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import ConfigurationSpace, EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from leanhebo_carps import timing
from leanhebo_carps.optimizer import LeanHEBOOptimizer

from leanhebo.space import Categorical as LeanCategorical
from leanhebo.space import Eq as LeanEq
from leanhebo.space import Float as LeanFloat
from leanhebo.space import Integer as LeanInteger


def _task(configspace: ConfigurationSpace, *, seed: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        name="test/task",
        seed=seed,
        input_space=SimpleNamespace(
            configuration_space=configspace,
            fidelity_space=SimpleNamespace(is_multifidelity=False),
        ),
        output_space=SimpleNamespace(n_objectives=1),
        optimization_resources=SimpleNamespace(n_trials=2, time_budget=None),
    )


def test_mixed_space_round_trip_and_incumbent() -> None:
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("rate", lower=1e-3, upper=1.0, log=True),
            UniformIntegerHyperparameter("depth", lower=1, upper=9, log=True),
            CategoricalHyperparameter("activation", ["relu", "tanh"]),
            OrdinalHyperparameter("width", [16, 32, 64]),
            Constant("fixed", "yes"),
        ]
    )
    optimizer = LeanHEBOOptimizer(_task(configspace), seed=7, leanhebo_config={})

    converted = {parameter.name: parameter for parameter in optimizer.leanhebo_space.parameters}
    assert isinstance(converted["rate"], LeanFloat) and converted["rate"].log
    assert isinstance(converted["depth"], LeanInteger) and converted["depth"].log
    assert isinstance(converted["activation"], LeanCategorical)
    assert isinstance(converted["width"], LeanInteger)
    assert (converted["width"].low, converted["width"].high) == (0, 2)
    assert isinstance(converted["fixed"], LeanCategorical)

    optimizer.setup_optimizer()
    first = optimizer.ask()
    second = optimizer.ask()
    assert first.config["width"] in (16, 32, 64)
    assert first.config["fixed"] == "yes"

    optimizer.tell(first, TrialValue(cost=2.0))
    optimizer.tell(second, TrialValue(cost=1.0))
    incumbent = optimizer.get_current_incumbent()
    assert incumbent is not None
    assert incumbent[0] == second
    assert incumbent[1].cost == 1.0


def test_conditional_space_round_trip() -> None:
    parent = CategoricalHyperparameter("model", ["a", "b"])
    child = UniformFloatHyperparameter("rate", lower=0.0, upper=1.0)
    configspace = ConfigurationSpace()
    configspace.add([parent, child, EqualsCondition(child, parent, "a")])

    optimizer = LeanHEBOOptimizer(_task(configspace), seed=7)
    converted = {parameter.name: parameter for parameter in optimizer.leanhebo_space.parameters}
    assert converted["rate"].active_when == LeanEq("model", "a")

    compiled = optimizer.leanhebo_space.compile()
    active = compiled.decode(compiled.encode([{"model": "a", "rate": 0.25}]))
    inactive = compiled.decode(compiled.encode([{"model": "b"}]))
    active_trial = optimizer.convert_to_trial(active)
    inactive_trial = optimizer.convert_to_trial(inactive)

    assert dict(active_trial.config) == {"model": "a", "rate": pytest.approx(0.25)}
    assert dict(inactive_trial.config) == {"model": "b"}


class _Logger:
    def __init__(self) -> None:
        self.rows: list[tuple[dict[str, object], str]] = []

    def log_arbitrary(self, data: dict[str, object], entity: str) -> None:
        self.rows.append((data, entity))


class _Optimizer(Optimizer):
    def _setup_optimizer(self) -> object:
        return object()

    def convert_configspace(self, configspace: object) -> object:
        return configspace

    def convert_to_trial(self, *args: tuple, **kwargs: dict) -> TrialInfo:
        return self.ask()

    def ask(self) -> TrialInfo:
        return TrialInfo(config={})  # type: ignore[arg-type]

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        pass

    def get_current_incumbent(self) -> None:
        return None


def test_timing_row_matches_carps_trial_numbering(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _Logger()
    wrapped = _Optimizer(_task(ConfigurationSpace()), loggers=[logger])  # type: ignore[list-item]
    optimizer = timing.TimedOptimizer(wrapped, "optimizer")
    ticks = iter((1.0, 1.25, 2.0, 2.75))
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))

    trial = optimizer.ask()
    optimizer.tell(trial, TrialValue(cost=1.0))

    assert logger.rows == [
        (
            {
                "optimizer_id": "optimizer",
                "task_id": "test/task",
                "seed": 7,
                "trial": 1,
                "ask_seconds": 0.25,
                "tell_seconds": 0.75,
            },
            "optimizer_timing",
        )
    ]


def test_upstream_conversion_does_not_print_and_setup_seeds_rngs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("hebo")
    from leanhebo_carps import upstream

    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", lower=-1.0, upper=1.0))
    optimizer = upstream.UpstreamHEBOOptimizer(_task(configspace), seed=13)
    trial = TrialInfo(config=configspace.get_default_configuration())
    seeds: list[tuple[str, int]] = []
    monkeypatch.setattr(upstream.np.random, "seed", lambda seed: seeds.append(("numpy", seed)))
    monkeypatch.setattr(upstream.torch, "manual_seed", lambda seed: seeds.append(("torch", seed)))
    monkeypatch.setattr(upstream.HEBOOptimizer, "_setup_optimizer", lambda self: "solver")

    frame = optimizer.convert_from_trial(trial)
    solver = optimizer._setup_optimizer()

    assert frame.to_dict(orient="records") == [{"x": 0.0}]
    assert solver == "solver"
    assert seeds == [("numpy", 13), ("torch", 13)]
    assert capsys.readouterr().out == ""
