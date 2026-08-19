"""Optimizer timing that is shared by LeanHEBO and upstream HEBO."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.types import Incumbent, SearchSpace
    from ConfigSpace import ConfigurationSpace


class TimedOptimizer(Optimizer):
    """Measure complete ask and tell calls without changing the CARP-S loop."""

    def __init__(self, optimizer: Optimizer, optimizer_id: str) -> None:
        super().__init__(
            optimizer.task,
            optimizer.loggers,
            expects_multiple_objectives=optimizer.expects_multiple_objectives,
            expects_fidelities=optimizer.expects_fidelities,
        )
        self.optimizer = optimizer
        self.optimizer_id = optimizer_id
        self._ask_seconds = 0.0
        self._trial = 0

    def _setup_optimizer(self) -> Any:
        self.optimizer.setup_optimizer()
        return self.optimizer.solver

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return self.optimizer.convert_configspace(configspace)

    def convert_to_trial(self, *args: Any, **kwargs: Any) -> TrialInfo:
        return self.optimizer.convert_to_trial(*args, **kwargs)

    def ask(self) -> TrialInfo:
        start = time.perf_counter()
        trial_info = self.optimizer.ask()
        self._ask_seconds = time.perf_counter() - start
        return trial_info

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        start = time.perf_counter()
        self.optimizer.tell(trial_info, trial_value)
        tell_seconds = time.perf_counter() - start
        self._trial += 1
        timing = {
            "optimizer_id": self.optimizer_id,
            "task_id": self.task.name,
            "seed": self.task.seed,
            "trial": self._trial,
            "ask_seconds": self._ask_seconds,
            "tell_seconds": tell_seconds,
        }
        for logger in self.loggers:
            logger.log_arbitrary(timing, "optimizer_timing")

    def get_current_incumbent(self) -> Incumbent:
        return self.optimizer.get_current_incumbent()


def timed_optimizer(
    optimizer: Callable[..., Optimizer],
    optimizer_id: str,
    task: Task,
    loggers: list[AbstractLogger] | None = None,
) -> TimedOptimizer:
    """Instantiate an optimizer and wrap its ask/tell calls with timing."""

    return TimedOptimizer(optimizer(task=task, loggers=loggers), optimizer_id)
