from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.carps.analyze import (
    Task,
    extract_curves,
    plot_curves,
    read_timing_jsonl,
    render_table,
)

TASKS = (
    Task(task_id="task-a", name="short-a", n_trials=2),
    Task(task_id="task-b", name="short-b", n_trials=4),
)
SEEDS = (1, 2)


def _results() -> tuple[pd.DataFrame, pd.DataFrame]:
    costs = {
        "LeanHEBO": {
            "task-a": [10.0, 6.0],
            "task-b": [10.0, 9.0, 8.0, 6.0],
        },
        "HEBO": {
            "task-a": [10.0, 8.0],
            "task-b": [10.0, 9.0, 8.5, 8.0],
        },
    }
    seconds = {"LeanHEBO": (0.10, 0.15), "HEBO": (0.40, 0.60)}
    logs: list[dict[str, object]] = []
    timing: list[dict[str, object]] = []
    for optimizer, task_costs in costs.items():
        for task_id, values in task_costs.items():
            for seed in SEEDS:
                for trial, cost in enumerate(values, 1):
                    logs.append(
                        {
                            "optimizer_id": optimizer,
                            "task_id": task_id,
                            "seed": seed,
                            "n_trials": trial,
                            "trial_value__cost": cost,
                            "trial_value__status": 1,
                        }
                    )
                    timing.append(
                        {
                            "optimizer_id": optimizer,
                            "task_id": f"short-{task_id[-1]}",
                            "seed": seed,
                            "trial": trial,
                            "ask_seconds": seconds[optimizer][0],
                            "tell_seconds": seconds[optimizer][1],
                        }
                    )
    return pd.DataFrame(logs), pd.DataFrame(timing)


def test_extracts_normalized_quality_and_optimizer_time() -> None:
    logs, timing = _results()

    curves = extract_curves(
        logs,
        timing,
        TASKS,
        seeds=SEEDS,
        fractions=pd.Series([0.5, 1.0]).to_numpy(),
    )

    assert curves.optimizers == ("HEBO", "LeanHEBO")
    assert curves.runs_per_optimizer == 4
    assert curves.quality[-1].tolist() == pytest.approx([0.5, 0.0])
    assert curves.optimizer_seconds[-1].tolist() == pytest.approx([3.0, 0.75])
    assert curves.quality_lower[-1].tolist() == pytest.approx([0.5, 0.0])
    table = render_table(curves, "HEBO")
    assert "| LeanHEBO | 0.0000 | -0.5000 [-0.5000, -0.5000] | 0.75 | 4.00x | 0 |" in table


def test_quality_interval_uses_seed_level_task_means() -> None:
    logs, timing = _results()
    for task in TASKS:
        selector = (
            (logs["optimizer_id"] == "HEBO")
            & (logs["task_id"] == task.task_id)
            & (logs["seed"] == 2)
            & (logs["n_trials"] == task.n_trials)
        )
        logs.loc[selector, "trial_value__cost"] = 6.0

    curves = extract_curves(
        logs,
        timing,
        TASKS,
        seeds=SEEDS,
        fractions=pd.Series([0.5, 1.0]).to_numpy(),
    )

    hebo = curves.optimizers.index("HEBO")
    assert curves.quality[-1, hebo] == pytest.approx(0.25)
    assert curves.quality_upper[-1, hebo] - curves.quality[-1, hebo] == pytest.approx(0.49)


def test_renders_two_panel_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    logs, timing = _results()
    curves = extract_curves(
        logs,
        timing,
        TASKS,
        seeds=SEEDS,
        fractions=pd.Series([0.5, 1.0]).to_numpy(),
    )
    output = tmp_path / "comparison.png"

    plot_curves(curves, output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_rejects_incomplete_run_matrix() -> None:
    logs, timing = _results()
    logs = logs.drop(logs.index[-1])

    with pytest.raises(ValueError, match="incomplete trials"):
        extract_curves(logs, timing, TASKS, seeds=SEEDS)


def test_rejects_timing_that_does_not_align_with_trials() -> None:
    logs, timing = _results()
    timing.loc[timing.index[-1], "trial"] = 5

    with pytest.raises(ValueError, match="incomplete trials"):
        extract_curves(logs, timing, TASKS, seeds=SEEDS)


def test_rejects_failed_trial() -> None:
    logs, timing = _results()
    logs.loc[logs.index[0], "trial_value__status"] = 2

    with pytest.raises(ValueError, match=r"failed trials \(LeanHEBO: 1\)"):
        extract_curves(logs, timing, TASKS, seeds=SEEDS)


def test_reads_timing_from_each_carps_run(tmp_path: Path) -> None:
    row = {
        "optimizer_id": "LeanHEBO",
        "task_id": "short-a",
        "seed": 1,
        "trial": 1,
        "ask_seconds": 0.1,
        "tell_seconds": 0.2,
    }
    for run in ("one", "two"):
        directory = tmp_path / run
        directory.mkdir()
        (directory / "optimizer_timing.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    timing = read_timing_jsonl(tmp_path)

    assert len(timing) == 2
