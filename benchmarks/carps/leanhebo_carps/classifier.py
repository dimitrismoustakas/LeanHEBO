"""Four-learner selection on fixed folds of scikit-learn classification datasets."""

import numpy as np
from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class ClassifierSelection(ObjectiveFunction):
    def __init__(self, dataset: str, seed: int) -> None:
        super().__init__()
        datasets = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits}
        x, y = datasets[dataset](return_X_y=True)
        self._samples = len(y)
        targets = np.eye(len(np.unique(y)))[y]
        self._folds = []
        splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=1147)
        for train, valid in splits.split(x, y):
            training, validation = x[train], x[valid]
            features = {"raw": (training, validation)}
            for name, scaler in (("standard", StandardScaler()), ("minmax", MinMaxScaler())):
                features[name] = (scaler.fit_transform(training), scaler.transform(validation))
            self._folds.append((features, y[train], targets[valid]))

        self._space = ConfigurationSpace(seed=seed)
        learner = Categorical("learner", ["logistic", "knn", "tree", "forest"])
        self._space.add(learner)
        for parameter, learners in (
            (Categorical("scaler", ["standard", "minmax"]), ["logistic", "knn"]),
            (Float("C", (1e-4, 1e4), log=True), ["logistic"]),
            (Categorical("intercept", [False, True]), ["logistic"]),
            (Integer("neighbors", (1, 32), log=True), ["knn"]),
            (Categorical("weights", ["distance", "uniform"]), ["knn"]),
            (Integer("p", (1, 2)), ["knn"]),
            (Integer("leaf", (1, 24), log=True), ["tree", "forest"]),
            (Integer("depth", (1, 20)), ["tree", "forest"]),
            (Categorical("criterion", ["gini", "entropy"]), ["tree", "forest"]),
            (Float("features", (0.1, 1.0)), ["forest"]),
        ):
            condition = (
                EqualsCondition(parameter, learner, learners[0])
                if len(learners) == 1
                else InCondition(parameter, learner, learners)
            )
            self._space.add([parameter, condition])

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._space

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        config = trial_info.config
        learner = config["learner"]
        scaler = config["scaler"] if learner in ("logistic", "knn") else "raw"
        total = 0.0
        for fold, (features, labels, targets) in enumerate(self._folds):
            train, valid = features[scaler]
            if learner == "logistic":
                estimator = LogisticRegression(
                    C=config["C"], fit_intercept=config["intercept"], max_iter=1500
                )
            elif learner == "knn":
                estimator = KNeighborsClassifier(
                    n_neighbors=config["neighbors"],
                    weights=config["weights"],
                    p=config["p"],
                    n_jobs=1,
                )
            else:
                tree_options = {
                    "min_samples_leaf": config["leaf"],
                    "max_depth": config["depth"],
                    "criterion": config["criterion"],
                    "random_state": 777 + fold,
                }
                if learner == "tree":
                    estimator = DecisionTreeClassifier(**tree_options)
                else:
                    estimator = RandomForestClassifier(
                        n_estimators=32, max_features=config["features"], n_jobs=1, **tree_options
                    )
            estimator.fit(train, labels)
            probabilities = estimator.predict_proba(valid)
            total += float(np.square(probabilities - targets).sum())
        return TrialValue(cost=total / self._samples)
