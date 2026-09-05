from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.carps.analyze import Run, load_runs, plot_curves, summarize


def _runs() -> list[Run]:
    runs = []
    for optimizer, last, seconds in (("LeanHEBO", 6.0, 0.25), ("HEBO", 8.0, 1.0)):
        for task, budget in (("task-a", 2), ("task-b", 4)):
            for seed in (1, 2):
                values = np.array([[10.0, seconds / 2, seconds / 2]] * budget)
                values[-1, 0] = last
                runs.append(Run(optimizer, task, seed, budget, values, False))
    return runs


def _write_runs(directory: Path, runs: list[Run]) -> None:
    for run in runs:
        path = directory / run.optimizer / run.task / str(run.seed) / "run.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "optimizer": run.optimizer,
                "task": run.task,
                "seed": run.seed,
                "n_trials": run.budget,
                "optimum": run.optimum,
                "metric": run.metric,
            }
        ]
        rows += [
            {"trial": i, "cost": cost, "ask_seconds": ask, "tell_seconds": tell}
            for i, (cost, ask, tell) in enumerate(run.values, 1)
        ]
        if run.failed:
            rows.append({"error": "optimizer failed"})
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_win_probabilities_native_costs_and_paired_timing():
    curves, probabilities, report = summarize(_runs(), "HEBO")
    final_probabilities = probabilities.groupby("fraction").mean().iloc[-1]
    assert final_probabilities.to_dict() == {"HEBO": 0.5, "LeanHEBO": 1.0}
    final = curves[(curves.fraction == 1) & (curves.optimizer == "LeanHEBO")]
    assert final.cost.tolist() == [6.0] * 4
    assert "| LeanHEBO | 100.0% |" in report
    assert "| 0.75 | 4.00x | 0/4 |" in report
    assert "| task-a | LeanHEBO | 6 | -2 |" in report


def test_failure_keeps_incumbent_and_excludes_same_timing_pair_for_every_optimizer(tmp_path):
    runs = _runs()
    failed = next(r for r in runs if r.optimizer == "HEBO" and r.task == "task-a" and r.seed == 2)
    failed.values = failed.values[:1]
    failed.failed = True
    _write_runs(tmp_path, runs)
    curves, _, report = summarize(load_runs([tmp_path]), "HEBO")
    pair = curves[(curves.task == "task-a") & (curves.seed == 2)]
    assert pair.seconds.isna().all()
    assert pair[(pair.optimizer == "HEBO") & (pair.fraction == 1)].cost.item() == 10.0
    assert "| HEBO | 50.0% |" in report
    assert "| 1/4 |" in report
    assert "| 1.00 | 4.00x | 0/4 |" in report


def test_outlier_optimizer_does_not_rescale_existing_costs_or_paired_differences():
    runs = _runs()
    before, before_probabilities, before_report = summarize(runs, "HEBO")
    extra = [
        Run("Outlier", r.task, r.seed, r.budget, r.values * 1000, False)
        for r in runs
        if r.optimizer == "HEBO"
    ]
    after, after_probabilities, after_report = summarize(runs + extra, "HEBO")
    np.testing.assert_array_equal(before.cost, after[after.optimizer != "Outlier"].cost)
    np.testing.assert_array_equal(
        before_probabilities, after_probabilities[before_probabilities.columns]
    )
    assert "| task-a | LeanHEBO | 6 | -2 |" in before_report
    assert "| task-a | LeanHEBO | 6 | -2 |" in after_report


def test_single_seed_and_arbitrary_task_count(tmp_path):
    runs = [run for run in _runs() if run.optimizer == "LeanHEBO" and run.seed == 1]
    _write_runs(tmp_path, runs)
    _, probabilities, _ = summarize(load_runs([tmp_path]), "LeanHEBO")
    assert (probabilities == 0.5).all().all()


def test_probability_uses_all_seed_pairs_half_ties_and_equal_task_weights():
    tasks = {
        "task-a": {"LeanHEBO": [1, 2], "HEBO": [2, 3]},
        "task-b": {"LeanHEBO": [5, 5, 5], "HEBO": [0, 0, 0]},
    }
    runs = [
        Run(optimizer, task, seed, 1, np.array([[cost, 1, 0]]), False)
        for task, optimizers in tasks.items()
        for optimizer, costs in optimizers.items()
        for seed, cost in enumerate(costs)
    ]
    _, probabilities, _ = summarize(runs, "HEBO")
    assert probabilities.loc[("task-a", 1.0), "LeanHEBO"] == 0.875
    assert probabilities.loc[("task-b", 1.0), "LeanHEBO"] == 0.0
    assert probabilities.groupby("fraction").mean().iloc[-1]["LeanHEBO"] == 0.4375


def test_rejects_missing_or_duplicate_runs(tmp_path):
    _write_runs(tmp_path, _runs()[:-1])
    with pytest.raises(ValueError, match="Missing 1"):
        load_runs([tmp_path])
    _write_runs(tmp_path, _runs())
    with pytest.raises(ValueError, match="Duplicate"):
        load_runs([tmp_path, tmp_path])


def test_rejects_nonconsecutive_trials_and_changed_objectives(tmp_path):
    _write_runs(tmp_path, _runs())
    path = next(tmp_path.rglob("run.jsonl"))
    text = path.read_text()
    path.write_text(text.replace('"trial": 2', '"trial": 3'))
    with pytest.raises(ValueError, match="Nonconsecutive"):
        load_runs([tmp_path])
    path.write_text(text.replace('"metric": "cost"', '"metric": "different"'))
    with pytest.raises(ValueError, match="Different budgets or objectives"):
        load_runs([tmp_path])


def test_writes_aggregate_and_task_figures(tmp_path):
    pytest.importorskip("matplotlib")
    runs = _runs()
    curves, probabilities, _ = summarize(runs, "HEBO")
    plot_curves(runs, curves, probabilities, "HEBO", tmp_path)
    for name in ("comparison.png", "tasks.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
