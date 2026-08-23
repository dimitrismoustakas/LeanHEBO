"""LeanHEBO's CARP-S optimizer adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.conditions import (
    AndConjunction,
    Conjunction,
    EqualsCondition,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    NotEqualsCondition,
    OrConjunction,
)
from ConfigSpace.conditions import (
    Condition as ConfigSpaceCondition,
)
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
from leanhebo.space import (
    All,
    Categorical,
    Condition,
    Eq,
    Float,
    GreaterThan,
    In,
    Integer,
    LessThan,
    NotEqual,
    Parameter,
    Space,
)
from leanhebo.space import (
    Any as AnyCondition,
)

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
        for hyperparameter in configspace.values():
            if isinstance(hyperparameter, OrdinalHyperparameter):
                self._ordinals[hyperparameter.name] = tuple(hyperparameter.sequence)

        parameters: list[Parameter] = []
        for hyperparameter in configspace.values():
            conditions = configspace.parent_conditions_of[hyperparameter.name]
            active_when = self._convert_conditions(conditions)
            parameters.append(self._convert_hyperparameter(hyperparameter, active_when))
        return Space(*parameters)

    def _convert_conditions(
        self,
        conditions: Sequence[ConfigSpaceCondition | Conjunction],
    ) -> Condition | None:
        return None if not conditions else self._convert_condition(conditions[0])

    def _convert_condition(
        self,
        condition: ConfigSpaceCondition | Conjunction,
    ) -> Condition:
        if isinstance(condition, AndConjunction):
            return All(*(self._convert_condition(child) for child in condition.components))
        if isinstance(condition, OrConjunction):
            return AnyCondition(*(self._convert_condition(child) for child in condition.components))

        parent = condition.parent.name
        if isinstance(condition, InCondition):
            return In(parent, (self._condition_value(parent, value) for value in condition.values))
        value = self._condition_value(parent, condition.value)
        if isinstance(condition, EqualsCondition):
            return Eq(parent, value)
        if isinstance(condition, NotEqualsCondition):
            return NotEqual(parent, value)
        if isinstance(condition, LessThanCondition):
            return LessThan(parent, value)
        if isinstance(condition, GreaterThanCondition):
            return GreaterThan(parent, value)
        raise TypeError(f"unsupported ConfigSpace condition: {type(condition).__name__}")

    def _condition_value(self, name: str, value: object) -> object:
        sequence = self._ordinals.get(name)
        return sequence.index(value) if sequence is not None else value

    def _convert_hyperparameter(
        self,
        hp: Hyperparameter,
        active_when: Condition | None,
    ) -> Parameter:
        if isinstance(hp, UniformFloatHyperparameter):
            return Float(
                hp.name,
                float(hp.lower),
                float(hp.upper),
                log=hp.log,
                active_when=active_when,
            )
        if isinstance(hp, UniformIntegerHyperparameter):
            return Integer(
                hp.name,
                int(hp.lower),
                int(hp.upper),
                log=hp.log,
                active_when=active_when,
            )
        if isinstance(hp, CategoricalHyperparameter):
            return Categorical(hp.name, tuple(hp.choices), active_when=active_when)
        if isinstance(hp, OrdinalHyperparameter):
            sequence = self._ordinals[hp.name]
            return Integer(hp.name, 0, len(sequence) - 1, active_when=active_when)
        if isinstance(hp, Constant):
            return Categorical(hp.name, (hp.value,), active_when=active_when)
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
            if name in values:
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
            if name in values:
                values[name] = sequence.index(values[name])
        cost = float(trial_value.cost)
        self.solver.observe([values], [cost])
        if self._incumbent is None or cost < float(self._incumbent[1].cost):
            self._incumbent = (trial_info, trial_value)

    def get_current_incumbent(self) -> Incumbent:
        return self._incumbent
