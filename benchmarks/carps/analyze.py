from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_RUN_KEYS = ["optimizer_id", "task_id", "seed"]
_ROW_KEYS = [*_RUN_KEYS, "n_trials"]
_LOG_COLUMNS = [*_ROW_KEYS, "trial_value__cost", "trial_value__status"]
_TIMING_COLUMNS = [*_RUN_KEYS, "trial", "ask_seconds", "tell_seconds"]
_SUCCESS = 1
_NORMALIZATION_EPSILON = 1e-8
_EXPECTED_TASKS = 13
DEFAULT_FRACTIONS = np.linspace(0.05, 1.0, 20)
DEFAULT_SEEDS = tuple(range(1, 21))


@dataclass(frozen=True)
class Curves:
    fractions: np.ndarray
    optimizers: tuple[str, ...]
    quality: np.ndarray
    quality_lower: np.ndarray
    quality_upper: np.ndarray
    quality_repeats: np.ndarray
    optimizer_seconds: np.ndarray
    runs_per_optimizer: int


@dataclass(frozen=True)
class Task:
    task_id: str
    name: str
    n_trials: int


def load_tasks(path: Path) -> tuple[Task, ...]:
    """Load the task IDs and budgets pinned by the benchmark protocol."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != _EXPECTED_TASKS:
        raise ValueError(f"tasks file must contain exactly {_EXPECTED_TASKS} tasks")

    tasks: list[Task] = []
    task_ids: set[str] = set()
    names: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("each task entry must be an object")
        task_id = entry.get("task_id")
        name = entry.get("name")
        budget = entry.get("n_trials")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("each task needs a non-empty task_id")
        if not isinstance(name, str) or not name:
            raise ValueError(f"task {task_id!r} needs a non-empty CARP-S name")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError(f"task {task_id!r} has an invalid n_trials budget")
        if task_id in task_ids:
            raise ValueError(f"duplicate task_id {task_id!r}")
        if name in names:
            raise ValueError(f"duplicate CARP-S task name {name!r}")
        task_ids.add(task_id)
        names.add(name)
        tasks.append(Task(task_id=task_id, name=name, n_trials=budget))
    return tuple(tasks)


def read_timing_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = sorted(path.rglob("optimizer_timing.jsonl")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"no optimizer_timing.jsonl files found under {path}")
    for timing_path in paths:
        with timing_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {timing_path} on line {line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"{timing_path} line {line_number} is not an object")
                rows.append(row)
    if not rows:
        raise ValueError("timing file is empty")
    return pd.DataFrame(rows)


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _integer_column(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError(f"{name} {column} must contain finite integers")
    return values.astype(np.int64)


def _numeric_column(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError(f"{name} {column} must contain finite numbers")
    return values.astype(float)


def _validate_matrix(
    logs: pd.DataFrame,
    timing: pd.DataFrame,
    tasks: tuple[Task, ...],
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    _require_columns(logs, _LOG_COLUMNS, "CARP-S logs")
    _require_columns(timing, _TIMING_COLUMNS, "timing data")
    logs = logs[_LOG_COLUMNS].copy()
    timing = timing[_TIMING_COLUMNS].copy()
    timing = timing.rename(columns={"trial": "n_trials"})

    budgets = {task.task_id: task.n_trials for task in tasks}
    task_names = {task.name: task.task_id for task in tasks}
    timing["task_id"] = timing["task_id"].map(task_names)

    for name, frame in (("CARP-S logs", logs), ("timing data", timing)):
        if frame[_ROW_KEYS].isnull().any(axis=None):
            raise ValueError(f"{name} contains missing run identifiers")
        frame["seed"] = _integer_column(frame, "seed", name)
        frame["n_trials"] = _integer_column(frame, "n_trials", name)
        frame["optimizer_id"] = frame["optimizer_id"].astype(str)
        frame["task_id"] = frame["task_id"].astype(str)
        if frame.duplicated(_ROW_KEYS).any():
            raise ValueError(f"{name} contains duplicate optimizer/task/seed/trial rows")

    logs["trial_value__cost"] = _numeric_column(logs, "trial_value__cost", "CARP-S logs")
    logs["trial_value__status"] = _integer_column(logs, "trial_value__status", "CARP-S logs")
    timing["ask_seconds"] = _numeric_column(timing, "ask_seconds", "timing data")
    timing["tell_seconds"] = _numeric_column(timing, "tell_seconds", "timing data")
    if (timing[["ask_seconds", "tell_seconds"]] < 0.0).any(axis=None):
        raise ValueError("timing data contains negative durations")

    failures = logs["trial_value__status"] != _SUCCESS
    if failures.any():
        counts = logs.loc[failures].groupby("optimizer_id", sort=True).size()
        detail = ", ".join(f"{optimizer}: {count}" for optimizer, count in counts.items())
        raise ValueError(f"CARP-S logs contain failed trials ({detail})")

    expected_tasks = set(budgets)
    actual_tasks = set(logs["task_id"])
    if actual_tasks != expected_tasks:
        missing = sorted(expected_tasks - actual_tasks)
        extra = sorted(actual_tasks - expected_tasks)
        raise ValueError(f"task matrix mismatch; missing={missing}, extra={extra}")

    optimizers = tuple(sorted(logs["optimizer_id"].unique().tolist()))
    if len(optimizers) < 2:
        raise ValueError("comparison needs at least two optimizers")
    if set(timing["optimizer_id"]) != set(optimizers):
        raise ValueError("quality and timing data contain different optimizers")
    if set(timing["task_id"]) != expected_tasks:
        raise ValueError("quality and timing data contain different tasks")

    expected_runs = {
        (optimizer, task_id, seed)
        for optimizer in optimizers
        for task_id in budgets
        for seed in seeds
    }
    for name, frame in (("CARP-S logs", logs), ("timing data", timing)):
        actual_runs = set(frame[_RUN_KEYS].itertuples(index=False, name=None))
        if actual_runs != expected_runs:
            missing_count = len(expected_runs - actual_runs)
            extra_count = len(actual_runs - expected_runs)
            raise ValueError(
                f"{name} run matrix is incomplete; missing={missing_count}, extra={extra_count}"
            )
        for (optimizer, task_id, seed), run in frame.groupby(_RUN_KEYS, sort=False):
            budget = budgets[task_id]
            actual_trials = np.sort(run["n_trials"].to_numpy())
            if not np.array_equal(actual_trials, np.arange(1, budget + 1)):
                raise ValueError(
                    f"{name} has incomplete trials for {optimizer}/{task_id}/seed={seed}"
                )

    return logs.sort_values(_ROW_KEYS), timing.sort_values(_ROW_KEYS), optimizers


def extract_curves(
    logs: pd.DataFrame,
    timing: pd.DataFrame,
    tasks: tuple[Task, ...],
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    fractions: np.ndarray = DEFAULT_FRACTIONS,
) -> Curves:
    """Validate complete runs and derive the two reported curves."""
    fractions = np.asarray(fractions, dtype=float)
    if (
        fractions.ndim != 1
        or fractions.size == 0
        or not np.all(np.isfinite(fractions))
        or np.any(fractions <= 0.0)
        or np.any(fractions > 1.0)
        or np.any(np.diff(fractions) <= 0.0)
        or fractions[-1] != 1.0
    ):
        raise ValueError("fractions must be increasing values in (0, 1] ending at 1")
    if not tasks:
        raise ValueError("task protocol is empty")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("expected seeds must contain at least two unique values")

    logs, timing, optimizers = _validate_matrix(logs, timing, tasks, seeds)

    # CARP-S normalizes each task over the results being compared. Recomputing
    # from raw costs means a later optimizer configuration needs analysis only.
    task_groups = logs.groupby("task_id")["trial_value__cost"]
    task_min = task_groups.transform("min")
    task_range = task_groups.transform("max") - task_min
    logs["normalized_cost"] = (logs["trial_value__cost"] - task_min) / (
        task_range + _NORMALIZATION_EPSILON
    )
    logs["incumbent"] = logs.groupby(_RUN_KEYS, sort=False)["normalized_cost"].cummin()
    timing["optimizer_seconds"] = timing["ask_seconds"] + timing["tell_seconds"]
    timing["optimizer_seconds"] = timing.groupby(_RUN_KEYS, sort=False)[
        "optimizer_seconds"
    ].cumsum()

    n_steps = fractions.size
    n_optimizers = len(optimizers)
    quality = np.empty((n_steps, n_optimizers))
    quality_lower = np.empty_like(quality)
    quality_upper = np.empty_like(quality)
    quality_repeats = np.empty((n_steps, len(seeds), n_optimizers))
    optimizer_seconds = np.empty_like(quality)

    for optimizer_index, optimizer in enumerate(optimizers):
        quality_samples: list[np.ndarray] = []
        timing_samples: list[np.ndarray] = []
        optimizer_logs = logs[logs["optimizer_id"] == optimizer]
        optimizer_timing = timing[timing["optimizer_id"] == optimizer]
        for task in tasks:
            task_id = task.task_id
            budget = task.n_trials
            indices = np.ceil(fractions * budget).astype(int) - 1
            for seed in seeds:
                run_selector = (optimizer_logs["task_id"] == task_id) & (
                    optimizer_logs["seed"] == seed
                )
                timing_selector = (optimizer_timing["task_id"] == task_id) & (
                    optimizer_timing["seed"] == seed
                )
                quality_samples.append(
                    optimizer_logs.loc[run_selector, "incumbent"].to_numpy()[indices]
                )
                timing_samples.append(
                    optimizer_timing.loc[timing_selector, "optimizer_seconds"].to_numpy()[indices]
                )

        quality_array = np.asarray(quality_samples).reshape(len(tasks), len(seeds), n_steps)
        timing_array = np.asarray(timing_samples)
        quality_by_seed = quality_array.mean(axis=0)
        quality_repeats[:, :, optimizer_index] = quality_by_seed.T
        quality[:, optimizer_index] = quality_by_seed.mean(axis=0)
        standard_error = quality_by_seed.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        quality_lower[:, optimizer_index] = quality[:, optimizer_index] - 1.96 * standard_error
        quality_upper[:, optimizer_index] = quality[:, optimizer_index] + 1.96 * standard_error
        optimizer_seconds[:, optimizer_index] = timing_array.mean(axis=0)

    return Curves(
        fractions=fractions,
        optimizers=optimizers,
        quality=quality,
        quality_lower=quality_lower,
        quality_upper=quality_upper,
        quality_repeats=quality_repeats,
        optimizer_seconds=optimizer_seconds,
        runs_per_optimizer=len(tasks) * len(seeds),
    )


def render_table(curves: Curves, reference: str) -> str:
    try:
        reference_index = curves.optimizers.index(reference)
    except ValueError as error:
        raise ValueError(f"reference optimizer {reference!r} is not present") from error

    reference_quality = curves.quality_repeats[-1, :, reference_index]
    reference_seconds = curves.optimizer_seconds[-1, reference_index]
    rows = [
        f"| Optimizer | Final normalized cost | Delta vs {reference} (95% CI) | Optimizer s | "
        f"Speed ratio ({reference}/optimizer) | Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index, optimizer in enumerate(curves.optimizers):
        delta_samples = curves.quality_repeats[-1, :, index] - reference_quality
        delta = delta_samples.mean()
        delta_half_width = 1.96 * delta_samples.std(ddof=1) / np.sqrt(delta_samples.size)
        seconds = curves.optimizer_seconds[-1, index]
        ratio = reference_seconds / seconds if seconds > 0.0 else np.inf
        ratio_text = f"{ratio:.2f}x" if np.isfinite(ratio) else "inf"
        rows.append(
            f"| {optimizer} | {curves.quality[-1, index]:.4f} | "
            f"{delta:+.4f} [{delta - delta_half_width:+.4f}, "
            f"{delta + delta_half_width:+.4f}] | {seconds:.2f} | "
            f"{ratio_text} | 0 |"
        )
    return "\n".join(rows)


def plot_curves(curves: Curves, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budget_percent = curves.fractions * 100.0
    figure, (quality_axis, time_axis) = plt.subplots(1, 2, figsize=(10, 4))
    for index, optimizer in enumerate(curves.optimizers):
        (line,) = quality_axis.plot(budget_percent, curves.quality[:, index], label=optimizer)
        quality_axis.fill_between(
            budget_percent,
            curves.quality_lower[:, index],
            curves.quality_upper[:, index],
            color=line.get_color(),
            alpha=0.15,
        )
        time_axis.plot(
            budget_percent,
            curves.optimizer_seconds[:, index],
            color=line.get_color(),
            label=optimizer,
        )

    quality_axis.set_title("Optimization quality")
    quality_axis.set_xlabel("Task budget used (%)")
    quality_axis.set_ylabel("Mean normalized incumbent cost (lower is better; 95% CI)")
    quality_axis.grid(alpha=0.25)
    quality_axis.legend()

    time_axis.set_title("Optimizer time")
    time_axis.set_xlabel("Task budget used (%)")
    time_axis.set_ylabel("Mean cumulative ask + tell (s)")
    time_axis.grid(alpha=0.25)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed LeanHEBO CARP-S comparison")
    parser.add_argument("--logs", type=Path, required=True, help="CARP-S logs.parquet")
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="CARP-S run directory containing optimizer_timing.jsonl files",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).with_name("tasks.json"),
        help="pinned CARP-S task protocol",
    )
    parser.add_argument("--reference", default="HEBO")
    parser.add_argument("--output", type=Path, default=Path("carps.png"))
    args = parser.parse_args(argv)

    if args.output.suffix.lower() != ".png":
        raise ValueError("--output must be a .png file")

    logs = pd.read_parquet(args.logs)
    timing = read_timing_jsonl(args.runs)
    curves = extract_curves(logs, timing, load_tasks(args.tasks))
    plot_curves(curves, args.output.resolve())

    print(
        f"Validated {len(curves.optimizers) * curves.runs_per_optimizer} completed runs; "
        "failed trials: 0.\n"
    )
    print(render_table(curves, args.reference))
    print(f"\nFigure: {args.output.resolve()}")


if __name__ == "__main__":
    main()
