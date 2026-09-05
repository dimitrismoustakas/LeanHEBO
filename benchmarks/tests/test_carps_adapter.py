from __future__ import annotations

import json
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
from leanhebo_carps.optimizer import LeanHEBOOptimizer
from leanhebo_carps.tasks import make_task
from omegaconf import OmegaConf

from benchmarks.carps.run import run_one
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


class _Optimizer(Optimizer):
    def _setup_optimizer(self) -> object:
        return object()

    def convert_configspace(self, configspace: object) -> object:
        return configspace

    def convert_to_trial(self, *args, **kwargs) -> TrialInfo:
        return self.ask()

    def ask(self) -> TrialInfo:
        return TrialInfo(
            config=self.task.input_space.configuration_space.get_default_configuration()
        )

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        raise RuntimeError("tell failed")

    def get_current_incumbent(self) -> None:
        return None


def test_runner_keeps_evaluated_point_and_timing_when_tell_fails(tmp_path, monkeypatch):
    import hydra.utils

    import benchmarks.carps.run as runner

    task = make_task("bbob/2/1/1", 2, 7)
    monkeypatch.setattr(hydra.utils, "instantiate", lambda *a, **kw: _Optimizer)
    ticks = iter((1.0, 1.25, 2.0, 2.75))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    output = tmp_path / "run.jsonl"
    assert not run_one(task, OmegaConf.create({}), "test", output)
    header, point, error = [json.loads(line) for line in output.read_text().splitlines()]
    assert header["n_trials"] == 2
    assert point["trial"] == 1
    assert point["ask_seconds"] == 0.25
    assert point["tell_seconds"] == 0.75
    assert set(point["config"]) == {"x0", "x1"}
    assert error == {"error": "RuntimeError: tell failed"}


def test_upstream_conversion_does_not_print_and_setup_seeds_rngs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("hebo")
    from leanhebo_carps import upstream

    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", lower=-1.0, upper=1.0))
    optimizer = upstream.UpstreamHEBOOptimizer(
        _task(configspace), seed=13, hebo_cfg=OmegaConf.create({"model_config": {"num_epochs": 1}})
    )
    assert type(optimizer.hebo_cfg["model_config"]) is dict
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


@pytest.mark.parametrize(
    "name", ["activation", "shared_root", "wide_exclusive", "threshold", "mixed"]
)
def test_synthetic_minimum_and_inactive_invariance(name):
    from ConfigSpace import Configuration

    task = make_task("synthetic/" + name, 3, 7)
    objective = task.objective_function
    optimum = Configuration(objective.configspace, values=objective.minimizer)
    assert objective.evaluate(TrialInfo(config=optimum)).cost == pytest.approx(0.0, abs=1e-20)
    for _ in range(12):
        sample = objective.configspace.sample_configuration()
        actual = objective.evaluate(TrialInfo(config=sample)).cost
        assert actual >= 0.0
        values = {key: hp.default_value for key, hp in objective.configspace.items()}
        values.update(dict(sample))
        dense = Configuration(objective.configspace, values=values, allow_inactive_with_values=True)
        assert objective.evaluate(TrialInfo(config=dense)).cost == actual
