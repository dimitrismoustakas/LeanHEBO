"""Compare saved runs using win probabilities, native costs, and paired optimizer time."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Run:
    optimizer: str
    task: str
    seed: int
    budget: int
    values: np.ndarray  # cost, ask seconds, tell seconds
    failed: bool
    optimum: float | None = None
    metric: str = "cost"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def read_run(path: Path) -> Run:
    header, *rows = read_jsonl(path)
    trials = [row for row in rows if "trial" in row]
    if [row["trial"] for row in trials] != list(range(1, len(trials) + 1)):
        raise ValueError(f"Nonconsecutive trials: {path}")
    values = np.asarray(
        [[row["cost"], row["ask_seconds"], row["tell_seconds"]] for row in trials], dtype=float
    ).reshape(-1, 3)
    return Run(
        header["optimizer"],
        header["task"],
        header["seed"],
        header["n_trials"],
        values,
        any("error" in row for row in rows) or len(trials) < header["n_trials"],
        header["optimum"],
        header["metric"],
    )


def read_carps_run(config_path: Path) -> Run:
    """Read the retained CARP-S comparisons without rewriting their raw results."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    task = OmegaConf.to_container(config.task, resolve=True)
    directory = config_path.parent.parent
    trial_path = directory / "trial_logs.jsonl"
    rows = read_jsonl(trial_path) if trial_path.exists() else []
    timing_path = directory / "optimizer_timing.jsonl"
    timing = read_jsonl(timing_path) if timing_path.exists() else []
    expected = list(range(1, len(rows) + 1))
    if [row["n_trials"] for row in rows] != expected or [
        row["trial"] for row in timing
    ] != expected:
        raise ValueError(f"Trial/timing mismatch: {directory}")
    values = np.asarray(
        [
            [row["trial_value"]["cost"], clock["ask_seconds"], clock["tell_seconds"]]
            for row, clock in zip(rows, timing, strict=True)
        ],
        dtype=float,
    ).reshape(-1, 3)
    succeeded = np.asarray([row["trial_value"]["status"] == 1 for row in rows])
    if len(values):
        values[~succeeded, 0] = np.inf
    optimum = None
    if task["name"].startswith("bbob/"):
        import ioh

        _, dimension, fid, instance = task["name"].split("/")
        optimum = float(ioh.get_problem(int(fid), int(instance), int(dimension)).optimum.y)
    return Run(
        config["optimizer_id"],
        task["name"],
        config["seed"],
        task["optimization_resources"]["n_trials"],
        values,
        not succeeded.all() or len(rows) < task["optimization_resources"]["n_trials"],
        optimum,
        task["output_space"]["objectives"][0],
    )


def load_runs(paths: list[Path]) -> list[Run]:
    runs = []
    for path in paths:
        runs.extend(read_run(p) for p in sorted(path.rglob("run.jsonl")))
        runs.extend(read_carps_run(p) for p in sorted(path.rglob(".hydra/config.yaml")))
    if not runs:
        raise ValueError("No runs found")
    keys = [(run.optimizer, run.task, run.seed) for run in runs]
    expected = {
        (o, t, s) for o in {r.optimizer for r in runs} for t, s in {(r.task, r.seed) for r in runs}
    }
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate optimizer/task/seed runs")
    if set(keys) != expected:
        raise ValueError(f"Missing {len(expected - set(keys))} optimizer/task/seed runs")
    for task in {run.task for run in runs}:
        if len({(r.budget, r.metric, r.optimum) for r in runs if r.task == task}) != 1:
            raise ValueError(f"Different budgets or objectives for {task}")
    for run in runs:
        if len(run.values) > run.budget or np.isnan(run.values).any():
            raise ValueError(f"Invalid trial data: {run.optimizer}/{run.task}/{run.seed}")
        if not np.isfinite(run.values[:, 1:]).all() or (run.values[:, 1:] < 0).any():
            raise ValueError(f"Invalid timing: {run.optimizer}/{run.task}/{run.seed}")
    return runs


def summarize(runs: list[Run], reference: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    failed_pairs = {(r.task, r.seed) for r in runs if r.failed}
    records = []
    for run in runs:
        costs = np.full(run.budget, np.inf)
        costs[: len(run.values)] = run.values[:, 0]
        incumbent = np.minimum.accumulate(costs)
        seconds = np.cumsum(run.values[:, 1:].sum(axis=1))
        for percent in range(5, 101, 5):
            index = (percent * run.budget - 1) // 100
            records.append(
                {
                    "optimizer": run.optimizer,
                    "task": run.task,
                    "seed": run.seed,
                    "fraction": percent / 100,
                    "cost": incumbent[index],
                    "seconds": seconds[index]
                    if (run.task, run.seed) not in failed_pairs
                    else np.nan,
                }
            )
    curves = pd.DataFrame(records)
    if reference not in curves.optimizer.unique():
        raise ValueError(f"Unknown reference optimizer: {reference}")
    comparisons = []
    for (task, fraction), group in curves.groupby(["task", "fraction"]):
        costs = group.pivot(index="seed", columns="optimizer", values="cost")
        baseline = costs[reference].to_numpy()
        for optimizer in costs.columns:
            own = costs[optimizer].to_numpy()[:, None]
            probability = np.mean(own < baseline) + 0.5 * np.mean(own == baseline)
            comparisons.append((task, fraction, optimizer, probability))
    probabilities = pd.DataFrame(
        comparisons, columns=["task", "fraction", "optimizer", "probability"]
    ).pivot(index=["task", "fraction"], columns="optimizer", values="probability")
    aggregate = probabilities.groupby("fraction").mean()
    final = curves[curves.fraction == 1.0]
    times = final.pivot(index=["task", "seed"], columns="optimizer", values="seconds")
    rows = [
        "# Optimizer comparison",
        "",
        f"Win probability compares every optimizer run with every {reference} run on the same "
        "task, counts ties as half, then averages tasks equally. 50% is neutral. "
        "Anytime probability averages 20 budget fractions from 5% to 100%. "
        "This measures how often an optimizer wins; task costs show the size of the differences.",
        "",
        "Failed runs retain their best evaluated cost for the remaining budget. "
        "Timing uses only task/seed pairs completed by every compared optimizer. "
        "Intervals in task plots are interquartile ranges, not confidence intervals.",
        "",
        f"| Optimizer | Final win vs {reference} | Anytime win vs {reference} | "
        "Median ask + tell (s) | "
        f"{reference}/optimizer | Failed runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for optimizer in probabilities.columns:
        own = [r for r in runs if r.optimizer == optimizer]
        seconds = times[optimizer].median()
        ratios = times[reference] / times[optimizer]
        speed = ratios.median()
        rows.append(
            f"| {optimizer} | {aggregate[optimizer].iloc[-1]:.1%} | "
            f"{aggregate[optimizer].mean():.1%} | "
            f"{seconds:.2f} | {speed:.2f}x | {sum(r.failed for r in own)}/{len(own)} |"
        )
    rows.extend(
        [
            "",
            "![Aggregate quality and optimizer time](comparison.png)",
            "",
            "![Per-task incumbent quality](tasks.png)",
            "",
            f"| Task | Optimizer | Median final cost | Median paired delta vs {reference} |",
            "|---|---|---:|---:|",
        ]
    )
    for task, group in final.groupby("task", sort=False):
        costs = group.pivot(index="seed", columns="optimizer", values="cost")
        for optimizer in probabilities.columns:
            rows.append(
                f"| {task} | {optimizer} | {costs[optimizer].median():.7g} | "
                f"{(costs[optimizer] - costs[reference]).median():+.7g} |"
            )
    return curves, probabilities, "\n".join(rows) + "\n"


def plot_curves(
    runs: list[Run], curves: pd.DataFrame, probabilities: pd.DataFrame, reference: str, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    tasks = list(dict.fromkeys(run.task for run in runs))
    aggregate = probabilities.groupby("fraction").mean()
    final = probabilities.xs(1.0, level="fraction").reindex(tasks)
    figure, axes = plt.subplots(1, 3, figsize=(16, max(4.5, 0.3 * len(tasks) + 1)))
    colors = {optimizer: f"C{i}" for i, optimizer in enumerate(probabilities.columns)}
    for optimizer in probabilities.columns:
        color = colors[optimizer]
        if optimizer != reference:
            axes[0].plot(100 * aggregate.index, aggregate[optimizer], color=color, label=optimizer)
            axes[0].annotate(
                f"{aggregate[optimizer].iloc[-1]:.1%}",
                (100, aggregate[optimizer].iloc[-1]),
                xytext=(-4, 10),
                textcoords="offset points",
                ha="right",
                color=color,
            )
            axes[1].hlines(range(len(tasks)), 0.5, final[optimizer], color=color, alpha=0.3)
            axes[1].scatter(final[optimizer], range(len(tasks)), color=color, s=30)
        time = curves[curves.optimizer == optimizer].groupby("fraction")["seconds"].median()
        axes[2].plot(100 * time.index, time, color=color, label=optimizer)
    axes[0].axhline(0.5, color="0.55", linestyle="--", linewidth=1)
    axes[0].set(
        ylim=(0, 1),
        xlim=(0, 103),
        ylabel=f"Probability of beating {reference}",
        xlabel="Evaluation budget used (%)",
        title=f"Across all {len(tasks)} tasks",
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axvline(0.5, color="0.55", linestyle="--", linewidth=1)
    axes[1].set(
        xlim=(0, 1),
        yticks=range(len(tasks)),
        yticklabels=[t.removeprefix("yahpo/").removesuffix("/None") for t in tasks],
        xlabel=f"Probability of beating {reference}",
        title="Each task at the full budget",
    )
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set(
        xlabel="Evaluation budget used (%)",
        ylabel="Median cumulative ask + tell (s)",
        title="Optimizer time",
    )
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)

    columns = min(3, len(tasks))
    figure, axes = plt.subplots(
        int(np.ceil(len(tasks) / columns)),
        columns,
        figsize=(4.5 * columns, 3 * np.ceil(len(tasks) / columns)),
        squeeze=False,
    )
    for axis, task in zip(axes.flat, tasks, strict=False):
        optimum = next(r.optimum for r in runs if r.task == task)
        for optimizer in probabilities.columns:
            selected = curves[(curves.task == task) & (curves.optimizer == optimizer)]
            values = selected.pivot(index="fraction", columns="seed", values="cost")
            if optimum is not None:
                values = (values - optimum).clip(lower=1e-12)
            axis.plot(
                100 * values.index, values.median(axis=1), color=colors[optimizer], label=optimizer
            )
            axis.fill_between(
                100 * values.index,
                values.quantile(0.25, axis=1),
                values.quantile(0.75, axis=1),
                color=colors[optimizer],
                alpha=0.15,
            )
        axis.set_title(task, fontsize=9)
        axis.set_xlabel("Budget (%)", fontsize=8)
        axis.set_ylabel("Regret" if optimum is not None else "Incumbent cost", fontsize=8)
        if optimum is not None:
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(tasks) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "tasks.png", dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", help="reference optimizer for paired differences and speed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_runs(args.runs)
    reference = args.reference or runs[0].optimizer
    curves, probabilities, report = summarize(runs, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_curves(runs, curves, probabilities, reference, args.output)
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(f"{len(runs)} runs, {sum(r.failed for r in runs)} failed; {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
