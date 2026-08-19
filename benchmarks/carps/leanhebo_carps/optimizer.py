"""LeanHEBO's CARP-S optimizer adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from leanhebo import LeanHEBO, LeanHEBOConfig
from leanhebo.data import CandidateBatch
from leanhebo.space import Categorical, Float, Integer, Parameter, Space

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent


class LeanHEBOOptimizer(Optimizer):
    """Expose LeanHEBO through CARP-S's sequential ask/tell contract."""

    def __init__(
        self,
        task: Task,
        seed: int,
        leanhebo_config: Mapping[str, Any] | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(task, loggers)
        if task.output_space.n_objectives != 1:
            raise ValueError("LeanHEBO supports single-objective tasks only")
        if task.input_space.fidelity_space.is_multifidelity:
            raise ValueError("LeanHEBO does not support multi-fidelity tasks")

        self.configspace = task.input_space.configuration_space
        if self.configspace.conditions:
            raise ValueError("LeanHEBO does not support conditional search spaces")
        if self.configspace.forbidden_clauses:
            raise ValueError("LeanHEBO does not support forbidden clauses")

        self._ordinals: dict[str, tuple[object, ...]] = {}
        self.leanhebo_space = self.convert_configspace(self.configspace)
        config = LeanHEBOConfig.from_dict({} if leanhebo_config is None else leanhebo_config)
        self.leanhebo_config = replace(
            config,
            runtime=replace(config.runtime, seed=seed),
            search=replace(config.search, seed=seed + 1),
        )
        self._incumbent: tuple[TrialInfo, TrialValue] | None = None

    def convert_configspace(self, configspace: ConfigurationSpace) -> Space:
        parameters: list[Parameter] = []
        for hyperparameter in configspace.values():
            parameters.append(self._convert_hyperparameter(hyperparameter))
        return Space(*parameters)

    def _convert_hyperparameter(self, hp: Hyperparameter) -> Parameter:
        if isinstance(hp, UniformFloatHyperparameter):
            return Float(hp.name, float(hp.lower), float(hp.upper), log=hp.log)
        if isinstance(hp, UniformIntegerHyperparameter):
            return Integer(hp.name, int(hp.lower), int(hp.upper), log=hp.log)
        if isinstance(hp, CategoricalHyperparameter):
            return Categorical(hp.name, tuple(hp.choices))
        if isinstance(hp, OrdinalHyperparameter):
            sequence = tuple(hp.sequence)
            self._ordinals[hp.name] = sequence
            return Integer(hp.name, 0, len(sequence) - 1)
        if isinstance(hp, Constant):
            return Categorical(hp.name, (hp.value,))
        raise TypeError(f"unsupported ConfigSpace parameter: {type(hp).__name__}")

    def _setup_optimizer(self) -> LeanHEBO:
        return LeanHEBO(self.leanhebo_space, config=self.leanhebo_config)

    def ask(self) -> TrialInfo:
        if not isinstance(self.solver, LeanHEBO):
            raise RuntimeError("optimizer has not been set up")
        return self.convert_to_trial(self.solver.suggest(1))

    def convert_to_trial(self, suggestion: CandidateBatch) -> TrialInfo:
        records = suggestion.to_records()
        if len(records) != 1:
            raise ValueError(f"CARP-S requires one suggestion, got {len(records)}")
        values = records[0]
        for name, sequence in self._ordinals.items():
            values[name] = sequence[cast(int, values[name])]
        config = Configuration(configuration_space=self.configspace, values=values)
        return TrialInfo(config=config)

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        if not isinstance(self.solver, LeanHEBO):
            raise RuntimeError("optimizer has not been set up")
        if isinstance(trial_value.cost, Sequence):
            raise ValueError("LeanHEBO supports one objective")

        values = dict(trial_info.config)
        for name, sequence in self._ordinals.items():
            values[name] = sequence.index(values[name])
        cost = float(trial_value.cost)
        self.solver.observe([values], [cost])
        if self._incumbent is None or cost < float(self._incumbent[1].cost):
            self._incumbent = (trial_info, trial_value)

    def get_current_incumbent(self) -> Incumbent:
        return self._incumbent
