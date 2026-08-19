"""Small correction to CARP-S's upstream HEBO adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from carps.optimizers.hebo import HEBOOptimizer
from carps.utils.trials import TrialInfo
from hebo.optimizers.hebo import HEBO

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from omegaconf import DictConfig


class UpstreamHEBOOptimizer(HEBOOptimizer):
    """Remove CARP-S's per-observation debug print from upstream HEBO timing."""

    def __init__(
        self,
        task: Task,
        seed: int,
        hebo_cfg: DictConfig | None = None,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,
        expects_fidelities: bool = False,
    ) -> None:
        self._seed = seed
        super().__init__(
            task=task,
            hebo_cfg=hebo_cfg,
            loggers=loggers,
            expects_multiple_objectives=expects_multiple_objectives,
            expects_fidelities=expects_fidelities,
        )

    def _setup_optimizer(self) -> HEBO:
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        return super()._setup_optimizer()

    def convert_from_trial(self, trial_info: TrialInfo) -> pd.DataFrame:
        return pd.DataFrame(dict(trial_info.config), index=[0])
