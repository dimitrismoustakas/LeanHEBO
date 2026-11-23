"""An exhaustively scored conditional grid of nearest neighbors and decision trees."""

from functools import cache
from itertools import product

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Integer

from leanhebo_carps.classifier import ClassifierSelection

_DOMAINS = {
    "knn": {
        "neighbors": range(1, 25),
        "scaler": ("standard", "minmax"),
        "weights": ("distance", "uniform"),
        "p": range(1, 3),
    },
    "tree": {
        "leaf": range(1, 9),
        "depth": range(1, 13),
        "criterion": ("gini", "entropy"),
    },
}


@cache
def _scores(dataset: str) -> dict[frozenset[tuple[str, object]], float]:
    objective = ClassifierSelection(dataset, seed=0)
    scores = {}
    for learner, parameters in _DOMAINS.items():
        for values in product(*parameters.values()):
            record = {"learner": learner, **dict(zip(parameters, values, strict=True))}
            trial = TrialInfo(config=Configuration(objective.configspace, values=record))
            scores[frozenset(record.items())] = float(objective.evaluate(trial).cost)
    return scores


class ClassifierGrid(ObjectiveFunction):
    def __init__(self, dataset: str, seed: int) -> None:
        super().__init__()
        self._space = ConfigurationSpace(seed=seed)
        learner = Categorical("learner", list(_DOMAINS))
        self._space.add(learner)
        for branch, parameters in _DOMAINS.items():
            for name, values in parameters.items():
                if isinstance(values, range):
                    parameter = Integer(
                        name, (values.start, values.stop - 1), log=name in {"neighbors", "leaf"}
                    )
                else:
                    parameter = Categorical(name, values)
                self._space.add([parameter, EqualsCondition(parameter, learner, branch)])
        self._scores = _scores(dataset)

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._space

    @property
    def f_min(self) -> float:
        return min(self._scores.values())

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return TrialValue(cost=self._scores[frozenset(trial_info.config.items())])
