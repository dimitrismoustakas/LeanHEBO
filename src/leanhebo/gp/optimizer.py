# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Optimizer implementations used by the exact GP."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast, overload

import torch


class PreconditionedSGLD(torch.optim.RMSprop):
    """RMSprop with the preconditioned Langevin perturbation used by HEBO."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        factor: float,
        pretrain_steps: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__(params, lr=lr, alpha=0.99, eps=1e-8, momentum=0, centered=False)
        self.factor = factor
        self.pretrain_steps = pretrain_steps
        self.steps = 0
        self.generator = generator

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = cast(float | None, super().step(closure))
        self.steps += 1
        if self.steps <= self.pretrain_steps:
            return loss
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            epsilon = float(group["eps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                square_average = self.state[parameter]["square_avg"]
                preconditioner = square_average.sqrt().add(epsilon)
                noise_scale = (2.0 * learning_rate / preconditioner).sqrt()
                noise = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=self.generator,
                )
                parameter.add_(noise * noise_scale * self.factor)
        return loss

    def set_observation_count(self, count: int) -> None:
        if count < 1:
            raise ValueError("observation count must be positive")
        self.factor = 1.0 / count

    def state_dict(self) -> dict[str, Any]:
        """Include the sampling schedule omitted by the base optimizer state."""

        state = super().state_dict()
        state["leanhebo_psgld"] = {
            "schema_version": 1,
            "factor": self.factor,
            "pretrain_steps": self.pretrain_steps,
            "steps": self.steps,
        }
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore custom state while accepting ordinary RMSprop state dictionaries."""

        base_state = dict(state_dict)
        custom = base_state.pop("leanhebo_psgld", None)
        super().load_state_dict(base_state)
        if custom is None:
            return
        if not isinstance(custom, Mapping) or int(custom.get("schema_version", -1)) != 1:
            raise ValueError("invalid PreconditionedSGLD state")
        factor = float(custom["factor"])
        pretrain_steps = int(custom["pretrain_steps"])
        steps = int(custom["steps"])
        if not math.isfinite(factor) or factor <= 0 or pretrain_steps < 0 or steps < 0:
            raise ValueError("invalid PreconditionedSGLD schedule state")
        self.factor = factor
        self.pretrain_steps = pretrain_steps
        self.steps = steps


def create_optimizer(
    name: str,
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    observations: int,
    pretrain_steps: int,
    lbfgs_max_iter: int,
    generator: torch.Generator,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "lbfgs":
        return torch.optim.LBFGS(
            parameters,
            lr=learning_rate,
            max_iter=lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
    if name == "psgld":
        return PreconditionedSGLD(
            parameters,
            lr=learning_rate,
            factor=1.0 / observations,
            pretrain_steps=pretrain_steps,
            generator=generator,
        )
    raise ValueError(f"unsupported GP optimizer: {name!r}")
