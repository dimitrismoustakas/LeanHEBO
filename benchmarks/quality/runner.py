# SPDX-License-Identifier: MIT

"""Run small, raw quality trials through either implementation's public API."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
    posterior_batch_size: int | None = 64
    gp_optimizer: Literal["psgld", "adam", "lbfgs"] = "psgld"
    learning_rate: float = 0.01
    model_lifecycle: Literal["cold", "persistent"] = "persistent"
    torch_threads: int = 1
    device: str = "cpu"
    dtype: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        for name in ("evaluation_budget", "batch_size", "random_samples", "population_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("generations", "gp_initial_steps", "gp_update_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.random_samples < 2:
            raise ValueError("random_samples must be at least two")
        if self.posterior_batch_size is not None and (
            isinstance(self.posterior_batch_size, bool)
            or not isinstance(self.posterior_batch_size, int)
            or self.posterior_batch_size < 1
        ):
            raise ValueError("posterior_batch_size must be positive or None")
        if self.gp_optimizer not in ("psgld", "adam", "lbfgs"):
            raise ValueError("unsupported GP optimizer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.model_lifecycle not in ("cold", "persistent"):
            raise ValueError("model_lifecycle must be cold or persistent")
        if (
            isinstance(self.torch_threads, bool)
            or not isinstance(self.torch_threads, int)
            or self.torch_threads < 1
        ):
            raise ValueError("torch_threads must be positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")

    @property
    def effective_update_steps(self) -> int:
        """Steps paid after the first fit under the selected lifecycle."""

        return self.gp_initial_steps if self.model_lifecycle == "cold" else self.gp_update_steps


@dataclass(frozen=True, slots=True)
class Suggested:
    rows: list[dict[str, object]]
    native: object
    search_report: Mapping[str, int] | None = None


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

        torch.set_num_threads(settings.torch_threads)
        space = _lean_space(objective.parameters)
        cold = settings.model_lifecycle == "cold"
        self._config = LeanHEBOConfig(
            random_samples=settings.random_samples,
            runtime=RuntimeConfig(
                device=settings.device,
                dtype=settings.dtype,
                seed=seed,
                acquisition_batch_size=settings.posterior_batch_size,
            ),
            gp=GPConfig(
                optimizer=settings.gp_optimizer,
                learning_rate=settings.learning_rate,
                initial_steps=settings.gp_initial_steps,
                update_steps=settings.effective_update_steps,
                full_refit_interval=1 if cold else None,
                full_refit_growth_factor=None,
                reuse_parameters=not cold,
                reuse_optimizer_state=not cold,
                use_set_train_data=not cold,
            ),
            search=SearchConfig(
                population_size=settings.population_size,
                generations=settings.generations,
                seed=seed + 1,
            ),
        )
        self._optimizer = LeanHEBO(space, config=self._config)
        self._torch = torch
        self._search_reports: list[dict[str, int]] = []
        self._implementation = {
            "name": "leanhebo",
            "version": leanhebo.__version__,
            **_local_revision(),
            "config": self._config.to_dict(),
        }
        self._work = WorkBudget(
            objective_evaluations=settings.evaluation_budget,
            batch_size=settings.batch_size,
            population_size=settings.population_size,
            generations=settings.generations,
            gp_initial_steps=settings.gp_initial_steps,
            gp_update_steps=settings.effective_update_steps,
            full_refit_interval=1 if cold else None,
            posterior_batch_size=settings.posterior_batch_size,
            random_samples=settings.random_samples,
            search_candidate_evaluations=(settings.population_size * (settings.generations + 1)),
            gp_optimizer=settings.gp_optimizer,
            learning_rate=settings.learning_rate,
            reuse_parameters=not cold,
            reuse_optimizer_state=not cold,
            use_set_train_data=not cold,
            device=settings.device,
            dtype=settings.dtype,
            torch_threads=settings.torch_threads,
        )

    @property
    def implementation(self) -> Mapping[str, object]:
        return self._implementation

    @property
    def work(self) -> WorkBudget:
        return self._work

    def suggest(self, count: int) -> Suggested:
        previous_search = self._optimizer.last_search
        candidates = self._optimizer.suggest(count)
        search = self._optimizer.last_search
        search_report: dict[str, int] | None = None
        if search is not None and search is not previous_search:
            search_report = {
                "objective_calls": search.objective_calls,
                "candidate_evaluations": search.candidate_evaluations,
                "offspring_generations": search.generations,
            }
            self._search_reports.append(search_report)
        return Suggested(candidates.to_records(), candidates, search_report)

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
            "search_reports": list(self._search_reports),
        }


class UpstreamHEBOAdapter:
    """Adapter loaded only inside the isolated upstream development environment."""

    _COMMIT = "ee6112d39d1a9e9703fecaf9057193e1ec9dae72"
    _POPULATION_SIZE = 100
    _PYMOO_GENERATIONS = 100
    _OFFSPRING_GENERATIONS = _PYMOO_GENERATIONS - 1

    def __init__(self, objective: ToyObjective, settings: RunSettings, seed: int) -> None:
        numpy = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        acquisition_module = importlib.import_module("hebo.acquisitions.acq")
        design_module = importlib.import_module("hebo.design_space.design_space")
        optimizer_module = importlib.import_module("hebo.optimizers.hebo")
        if settings.device != "cpu" or settings.dtype != "float32":
            raise ValueError("the pinned upstream HEBO lane supports only CPU float32")
        if settings.model_lifecycle != "cold":
            raise ValueError("the pinned upstream HEBO implementation always uses a cold model")
        torch.set_num_threads(settings.torch_threads)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        design_space = design_module.DesignSpace().parse(
            [parameter.to_upstream_spec() for parameter in objective.parameters]
        )
        model_config = {
            "lr": settings.learning_rate,
            "optimizer": settings.gp_optimizer,
            "num_epochs": settings.gp_initial_steps,
            "verbose": False,
            "noise_lb": 8e-4,
            "pred_likeli": False,
        }
        self._active_search = {"objective_calls": 0, "candidate_evaluations": 0}
        self._search_reports: list[dict[str, int]] = []
        self._mace_class = acquisition_module.MACE

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
                "search": {
                    "population_size": self._POPULATION_SIZE,
                    "pymoo_generations": self._PYMOO_GENERATIONS,
                    "offspring_generations": self._OFFSPRING_GENERATIONS,
                },
            },
        }
        # The audited implementation reconstructs and fully fits the GP for every model-based
        # suggestion. Its acquisition search hard-codes 100 by 100.
        self._work = WorkBudget(
            objective_evaluations=settings.evaluation_budget,
            batch_size=settings.batch_size,
            population_size=self._POPULATION_SIZE,
            generations=self._OFFSPRING_GENERATIONS,
            gp_initial_steps=settings.gp_initial_steps,
            gp_update_steps=settings.gp_initial_steps,
            full_refit_interval=1,
            posterior_batch_size=None,
            random_samples=settings.random_samples,
            search_candidate_evaluations=(
                self._POPULATION_SIZE * (self._OFFSPRING_GENERATIONS + 1)
            ),
            gp_optimizer=settings.gp_optimizer,
            learning_rate=settings.learning_rate,
            reuse_parameters=False,
            reuse_optimizer_state=False,
            use_set_train_data=False,
            device="cpu",
            dtype="float32",
            torch_threads=settings.torch_threads,
        )

    @property
    def implementation(self) -> Mapping[str, object]:
        return self._implementation

    @property
    def work(self) -> WorkBudget:
        return self._work

    def suggest(self, count: int) -> Suggested:
        model_based = int(self._optimizer.X.shape[0]) >= int(self._work.random_samples or 0)
        self._active_search = {"objective_calls": 0, "candidate_evaluations": 0}
        original_eval = self._mace_class.eval
        adapter = self

        def counted_eval(acquisition: object, x: object, xe: object) -> object:
            rows = int(x.shape[0])  # type: ignore[attr-defined]
            adapter._active_search["objective_calls"] += 1
            adapter._active_search["candidate_evaluations"] += rows
            return original_eval(acquisition, x, xe)

        self._mace_class.eval = counted_eval
        try:
            candidates = self._optimizer.suggest(count)
        finally:
            self._mace_class.eval = original_eval
        if model_based:
            search_report = {
                **self._active_search,
                "offspring_generations": self._OFFSPRING_GENERATIONS,
            }
            self._search_reports.append(search_report)
        else:
            search_report = None
        rows = candidates.to_dict(orient="records")
        return Suggested([dict(row) for row in rows], candidates, search_report)

    def observe(self, suggested: Suggested, values: Sequence[float]) -> None:
        outcomes = self._numpy.asarray(values, dtype=float).reshape(-1, 1)
        self._optimizer.observe(suggested.native, outcomes)

    def phase_wall_times(self) -> Mapping[str, Sequence[float]]:
        return {}

    def metrics(self) -> Mapping[str, object]:
        return {
            "observations": int(self._optimizer.y.shape[0]),
            "search_reports": list(self._search_reports),
        }


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


def _local_revision() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    commit_process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = commit_process.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("could not resolve a full lowercase LeanHEBO Git commit")
    status_process = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "repository": str(repository),
        "commit": commit,
        "source_dirty": bool(status_process.stdout.strip()),
    }


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
    duplicate_suggestions = 0
    seen_candidates: set[str] = set()
    first_model_suggestion_seen = False

    while len(observed) < adapter.work.objective_evaluations:
        requested = min(
            adapter.work.batch_size,
            adapter.work.objective_evaluations - len(observed),
        )
        stage = "suggest"
        try:
            if len(observed) < int(adapter.work.random_samples or 0):
                suggest_phase = "driver.suggest.initial_sobol"
            elif not first_model_suggestion_seen:
                suggest_phase = "driver.suggest.first_model"
                first_model_suggestion_seen = True
            else:
                suggest_phase = "driver.suggest.steady_model"
            with recorder.phase(suggest_phase):
                suggested = adapter.suggest(requested)
            if len(suggested.rows) != requested:
                invalid_suggestions += abs(requested - len(suggested.rows))
                raise RuntimeError(
                    f"optimizer returned {len(suggested.rows)} candidates; expected {requested}"
                )
            stage = "search-work"
            _validate_search_work(
                suggested.search_report,
                adapter.work,
                model_based=suggest_phase != "driver.suggest.initial_sobol",
            )
            for row in suggested.rows:
                key = json.dumps(
                    [row[parameter.name] for parameter in objective.parameters],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if key in seen_candidates:
                    duplicate_suggestions += 1
                else:
                    seen_candidates.add(key)
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
        "duplicate_suggestions": duplicate_suggestions,
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


def _validate_search_work(
    report: Mapping[str, int] | None,
    work: WorkBudget,
    *,
    model_based: bool,
) -> None:
    declared_candidates = work.search_candidate_evaluations
    if not model_based:
        if report is not None:
            raise RuntimeError("initial Sobol suggestion unexpectedly reported acquisition search")
        return
    if declared_candidates is None:
        return
    if report is None:
        raise RuntimeError("model-based suggestion did not report actual search work")
    actual_candidates = report.get("candidate_evaluations")
    if actual_candidates != declared_candidates:
        raise RuntimeError(
            "search candidate work differed from its declaration: "
            f"actual={actual_candidates}, declared={declared_candidates}"
        )
    if work.generations is not None:
        expected_calls = work.generations + 1
        actual_calls = report.get("objective_calls")
        if actual_calls != expected_calls:
            raise RuntimeError(
                "search objective-call work differed from its declaration: "
                f"actual={actual_calls}, declared={expected_calls}"
            )
