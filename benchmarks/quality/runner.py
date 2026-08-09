# SPDX-License-Identifier: MIT

"""Run small, raw quality trials through either implementation's public API."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from benchmarks.harness.results import BenchmarkResult, PhaseRecorder
from benchmarks.harness.work import WorkBudget
from benchmarks.quality.objectives import ParameterDefinition, ToyObjective

if TYPE_CHECKING:
    from leanhebo.space import Space


@dataclass(frozen=True, slots=True)
class RunSettings:
    evaluation_budget: int = 8
    batch_size: int = 2
    random_samples: int = 4
    population_size: int = 12
    generations: int = 2
    gp_initial_steps: int = 2
    gp_update_steps: int = 1
    posterior_batch_size: int = 64
    device: str = "cpu"
    dtype: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        for name in ("evaluation_budget", "batch_size", "random_samples", "population_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("generations", "gp_initial_steps", "gp_update_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.posterior_batch_size < 1:
            raise ValueError("posterior_batch_size must be positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


@dataclass(frozen=True, slots=True)
class Suggested:
    rows: list[dict[str, object]]
    native: object


class OptimizerAdapter(Protocol):
    @property
    def implementation(self) -> Mapping[str, object]: ...

    @property
    def work(self) -> WorkBudget: ...

    def suggest(self, count: int) -> Suggested: ...

    def observe(self, suggested: Suggested, values: Sequence[float]) -> None: ...

    def phase_wall_times(self) -> Mapping[str, Sequence[float]]: ...

    def metrics(self) -> Mapping[str, object]: ...


class LeanHEBOAdapter:
    def __init__(self, objective: ToyObjective, settings: RunSettings, seed: int) -> None:
        import torch

        import leanhebo
        from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig
        from leanhebo.optimizer import LeanHEBO

        space = _lean_space(objective.parameters)
        self._config = LeanHEBOConfig(
            random_samples=settings.random_samples,
            runtime=RuntimeConfig(
                device=settings.device,
                dtype=settings.dtype,
                seed=seed,
                acquisition_batch_size=settings.posterior_batch_size,
            ),
            gp=GPConfig(
                optimizer="adam",
                initial_steps=settings.gp_initial_steps,
                update_steps=settings.gp_update_steps,
                full_refit_interval=None,
                full_refit_growth_factor=None,
            ),
            search=SearchConfig(
                population_size=settings.population_size,
                generations=settings.generations,
                seed=seed + 1,
            ),
        )
        self._optimizer = LeanHEBO(space, config=self._config)
        self._torch = torch
        self._implementation = {
            "name": "leanhebo",
            "version": leanhebo.__version__,
            "repository": None,
            "commit": None,
            "config": self._config.to_dict(),
        }
        self._work = WorkBudget(
            objective_evaluations=settings.evaluation_budget,
            batch_size=settings.batch_size,
            population_size=settings.population_size,
            generations=settings.generations,
            gp_initial_steps=settings.gp_initial_steps,
            gp_update_steps=settings.gp_update_steps,
            full_refit_interval=None,
            posterior_batch_size=settings.posterior_batch_size,
            random_samples=settings.random_samples,
        )

    @property
    def implementation(self) -> Mapping[str, object]:
        return self._implementation

    @property
    def work(self) -> WorkBudget:
        return self._work

    def suggest(self, count: int) -> Suggested:
        candidates = self._optimizer.suggest(count)
        return Suggested(candidates.to_records(), candidates)

    def observe(self, suggested: Suggested, values: Sequence[float]) -> None:
        outcomes = self._torch.tensor(values, dtype=self._optimizer.dtype)
        retained = self._optimizer.observe(suggested.native, outcomes)
        if retained != len(values):
            raise RuntimeError(f"LeanHEBO retained {retained} of {len(values)} finite outcomes")

    def phase_wall_times(self) -> Mapping[str, Sequence[float]]:
        return self._optimizer.diagnostics.phase_seconds

    def metrics(self) -> Mapping[str, object]:
        diagnostics = self._optimizer.diagnostics
        surrogate = self._optimizer.surrogate
        return {
            "diagnostic_counters": dict(diagnostics.counters),
            "fit_reports": [
                {
                    "kind": report.kind,
                    "observations": report.observations,
                    "requested_steps": report.requested_steps,
                    "completed_steps": report.completed_steps,
                    "maximum_jitter": report.maximum_jitter,
                    "jitter_retries": report.jitter_retries,
                    "failure": report.failure,
                }
                for report in diagnostics.fit_reports
            ],
            "posterior_calls": None if surrogate is None else surrogate.posterior_calls,
        }


class UpstreamHEBOAdapter:
    """Adapter loaded only inside the isolated upstream development environment."""

    _COMMIT = "ee6112d39d1a9e9703fecaf9057193e1ec9dae72"

    def __init__(self, objective: ToyObjective, settings: RunSettings, seed: int) -> None:
        numpy = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        design_module = importlib.import_module("hebo.design_space.design_space")
        optimizer_module = importlib.import_module("hebo.optimizers.hebo")
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        design_space = design_module.DesignSpace().parse(
            [parameter.to_upstream_spec() for parameter in objective.parameters]
        )
        model_config = {
            "lr": 0.01,
            "num_epochs": settings.gp_initial_steps,
            "verbose": False,
            "noise_lb": 8e-4,
            "pred_likeli": False,
        }
        self._optimizer = optimizer_module.HEBO(
            design_space,
            rand_sample=settings.random_samples,
            model_config=model_config,
            scramble_seed=seed,
        )
        self._numpy = numpy
        self._implementation = {
            "name": "upstream-hebo",
            "version": "0.3.6",
            "repository": "https://github.com/huawei-noah/HEBO.git",
            "commit": self._COMMIT,
            "config": {
                "rand_sample": settings.random_samples,
                "model_config": model_config,
                "search": {"population_size": 100, "generations": 100},
            },
        }
        # The audited implementation reconstructs and fully fits the GP for every model-based
        # suggestion. Its acquisition search hard-codes 100 by 100.
        self._work = WorkBudget(
            objective_evaluations=settings.evaluation_budget,
            batch_size=settings.batch_size,
            population_size=100,
            generations=100,
            gp_initial_steps=settings.gp_initial_steps,
            gp_update_steps=settings.gp_initial_steps,
            full_refit_interval=1,
            posterior_batch_size=None,
            random_samples=settings.random_samples,
        )

    @property
    def implementation(self) -> Mapping[str, object]:
        return self._implementation

    @property
    def work(self) -> WorkBudget:
        return self._work

    def suggest(self, count: int) -> Suggested:
        candidates = self._optimizer.suggest(count)
        rows = candidates.to_dict(orient="records")
        return Suggested([dict(row) for row in rows], candidates)

    def observe(self, suggested: Suggested, values: Sequence[float]) -> None:
        outcomes = self._numpy.asarray(values, dtype=float).reshape(-1, 1)
        self._optimizer.observe(suggested.native, outcomes)

    def phase_wall_times(self) -> Mapping[str, Sequence[float]]:
        return {}

    def metrics(self) -> Mapping[str, object]:
        return {"observations": int(self._optimizer.y.shape[0])}


def _lean_space(parameters: Sequence[ParameterDefinition]) -> Space:
    from leanhebo.space import Categorical, Float, Integer, Parameter, Space

    converted: list[Parameter] = []
    for parameter in parameters:
        if parameter.kind == "categorical":
            converted.append(Categorical(parameter.name, parameter.categories))
        elif parameter.kind == "integer":
            assert isinstance(parameter.lower, int) and isinstance(parameter.upper, int)
            converted.append(Integer(parameter.name, parameter.lower, parameter.upper))
        else:
            assert parameter.lower is not None and parameter.upper is not None
            converted.append(Float(parameter.name, float(parameter.lower), float(parameter.upper)))
    return Space(*converted)


def make_adapter(
    implementation: str,
    objective: ToyObjective,
    settings: RunSettings,
    seed: int,
) -> OptimizerAdapter:
    if implementation == "leanhebo":
        return LeanHEBOAdapter(objective, settings, seed)
    if implementation == "upstream-hebo":
        return UpstreamHEBOAdapter(objective, settings, seed)
    raise ValueError(f"unsupported implementation: {implementation}")


def run_trial(adapter: OptimizerAdapter, objective: ToyObjective, seed: int) -> BenchmarkResult:
    """Run one trial and preserve failures in the raw record instead of dropping the seed."""

    recorder = PhaseRecorder()
    observed: list[float] = []
    best_so_far: list[float] = []
    failures: list[dict[str, object]] = []
    invalid_suggestions = 0

    while len(observed) < adapter.work.objective_evaluations:
        requested = min(
            adapter.work.batch_size,
            adapter.work.objective_evaluations - len(observed),
        )
        stage = "suggest"
        try:
            with recorder.phase("driver.suggest"):
                suggested = adapter.suggest(requested)
            if len(suggested.rows) != requested:
                invalid_suggestions += abs(requested - len(suggested.rows))
                raise RuntimeError(
                    f"optimizer returned {len(suggested.rows)} candidates; expected {requested}"
                )
            stage = "objective"
            with recorder.phase("driver.objective"):
                values = objective.evaluate(suggested.rows)
            if len(values) != requested or any(not math.isfinite(value) for value in values):
                raise RuntimeError("objective evaluation returned an invalid batch")
            stage = "observe"
            with recorder.phase("driver.observe"):
                adapter.observe(suggested, values)
        except Exception as error:  # A quality suite must report failed seeds, not omit them.
            failures.append(
                {"stage": stage, "message": str(error), "exception_type": type(error).__name__}
            )
            break

        for value in values:
            observed.append(value)
            best_so_far.append(value if not best_so_far else min(best_so_far[-1], value))

    recorder.merge_wall_times(adapter.phase_wall_times())
    final_best = None if not best_so_far else best_so_far[-1]
    normalized_regret = None if final_best is None else objective.normalized_regret(final_best)
    metrics: dict[str, object] = {
        "evaluations_completed": len(observed),
        "invalid_suggestions": invalid_suggestions,
        "failed": bool(failures),
        "implementation_metrics": adapter.metrics(),
    }
    quality: dict[str, object] = {
        "objective": objective.name,
        "optimum": objective.optimum,
        "regret_scale": objective.regret_scale,
        "objective_values": observed,
        "best_so_far": best_so_far,
        "final_best": final_best,
        "normalized_regret": normalized_regret,
    }
    return BenchmarkResult(
        implementation=adapter.implementation,
        suite="toy-quality",
        case=objective.name,
        seed=seed,
        work=adapter.work,
        phases=recorder.to_dict(),
        metrics=metrics,
        quality=quality,
        failures=failures,
    )
