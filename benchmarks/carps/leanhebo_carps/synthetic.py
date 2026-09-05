"""Conditional quadratics with analytic minimum zero.

Every objective is a nonnegative branch offset plus positively weighted squares.
The stored minimizer activates a zero-offset branch and makes every square zero.
These diagnose conditional search; they are not surrogates for real model training.
"""

from math import log10

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    GreaterThanCondition,
    InCondition,
    Integer,
)


class ConditionalQuadratic(ObjectiveFunction):
    def __init__(self, name: str, seed: int = 0) -> None:
        super().__init__()
        self.name = name
        self._space = ConfigurationSpace(seed=seed)
        if name in ("activation", "shared_root", "wide_exclusive"):
            self.offsets, self.roots = {
                "activation": ((0.12, 0.0), (0.17, 0.71, 0.43)),
                "shared_root": ((0.12, 0.04, 0.0, 0.08, 0.16, 0.10), (0.19, 0.73, 0.41)),
                "wide_exclusive": ((0.09, 0.14, 0.05, 0.11, 0.0, 0.07, 0.16, 0.03), ()),
            }[name]
            branch = Categorical("branch", list(range(len(self.offsets))))
            self._space.add(branch)
            self._space.add([Float(f"root_{i}", (0.0, 1.0)) for i in range(len(self.roots))])
            self.targets = {}
            for b in range(len(self.offsets)):
                if name == "activation" and b == 0:
                    continue
                targets = tuple(((b * 7 + i * 3 + 2) % 17 + 1) / 18 for i in range(3))
                self.targets[b] = targets
                for i in range(3):
                    child = Float(f"child_{b}_{i}", (0.0, 1.0))
                    self._space.add([child, EqualsCondition(child, branch, b)])
            best = self.offsets.index(0.0)
            self.minimizer = {
                "branch": best,
                **{f"root_{i}": target for i, target in enumerate(self.roots)},
                **{f"child_{best}_{i}": target for i, target in enumerate(self.targets[best])},
            }
        elif name == "threshold":
            gate = Float("gate", (0.0, 1.0))
            self._space.add([gate, Float("root_0", (0.0, 1.0)), Float("root_1", (0.0, 1.0))])
            for i in range(2):
                child = Float(f"child_{i}", (0.0, 1.0))
                self._space.add([child, GreaterThanCondition(child, gate, 0.9)])
            self.minimizer = {
                "gate": 0.965,
                "root_0": 0.23,
                "root_1": 0.76,
                "child_0": 0.18,
                "child_1": 0.83,
            }
        elif name == "mixed":
            branch = Categorical("branch", ["linear", "tree", "dropout"])
            self._space.add(branch)
            self.parameters = {
                "regularization": (1e-4, 10.0, 0.025, True, None),
                "rate": (0.01, 0.3, 0.075, True, ["tree", "dropout"]),
                "rounds": (50, 500, 240, True, ["tree", "dropout"]),
                "depth": (2, 12, 6, False, ["tree", "dropout"]),
                "drop": (0.0, 0.5, 0.12, False, ["dropout"]),
            }
            for key, (low, high, _, log, branches) in self.parameters.items():
                parameter_type = Integer if key in ("rounds", "depth") else Float
                hp = parameter_type(key, (low, high), log=log)
                self._space.add(hp)
                if branches is not None:
                    self._space.add(InCondition(hp, branch, branches))
            self.minimizer = {"branch": "dropout", **{k: p[2] for k, p in self.parameters.items()}}
        else:
            raise ValueError(f"Unknown conditional quadratic: {name}")

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._space

    @property
    def f_min(self) -> float:
        return 0.0

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        x = dict(trial_info.config)
        if self.name == "threshold":
            value = ((x["root_0"] - 0.23) ** 2 + (x["root_1"] - 0.76) ** 2) / 2
            if x["gate"] <= 0.9:
                value += 0.10 + 0.25 * (x["gate"] - 0.9) ** 2
            else:
                value += (x["gate"] - 0.965) ** 2
                value += ((x["child_0"] - 0.18) ** 2 + (x["child_1"] - 0.83) ** 2) / 2
        elif self.name == "mixed":
            value = {"linear": 0.15, "tree": 0.04, "dropout": 0.0}[x["branch"]]
            for key, (low, high, target, log, branches) in self.parameters.items():
                if branches is None or x["branch"] in branches:
                    delta = (
                        log10(x[key] / target) / log10(high / low)
                        if log
                        else (x[key] - target) / (high - low)
                    )
                    value += delta**2
        else:
            branch = int(x["branch"])
            value = self.offsets[branch]
            if self.roots:
                weight = 2.0 if self.name == "shared_root" else 1.0
                value += (
                    weight
                    * sum((x[f"root_{i}"] - target) ** 2 for i, target in enumerate(self.roots))
                    / len(self.roots)
                )
            for i, target in enumerate(self.targets.get(branch, ())):
                value += (x[f"child_{branch}_{i}"] - target) ** 2 / 3
        return TrialValue(cost=float(value))
