# SPDX-License-Identifier: MIT

"""Capture one reproducible cProfile trace and its matching raw quality result."""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path

from benchmarks.harness.results import write_result
from benchmarks.quality.objectives import get_objective
from benchmarks.quality.runner import RunSettings, make_adapter, run_trial


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sphere-2d")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-budget", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--random-samples", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--gp-initial-steps", type=int, default=2)
    parser.add_argument("--gp-update-steps", type=int, default=1)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "profiles",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    objective = get_objective(args.case)
    settings = RunSettings(
        evaluation_budget=args.evaluation_budget,
        batch_size=args.batch_size,
        random_samples=args.random_samples,
        population_size=args.population_size,
        generations=args.generations,
        gp_initial_steps=args.gp_initial_steps,
        gp_update_steps=args.gp_update_steps,
    )
    adapter = make_adapter("leanhebo", objective, settings, args.seed)
    profiler = cProfile.Profile()
    profiler.enable()
    result = run_trial(adapter, objective, args.seed)
    profiler.disable()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_directory / f"{result.run_id}.prof"
    result_path = args.output_directory / f"{result.run_id}.json"
    if profile_path.exists():
        raise FileExistsError(f"refusing to overwrite profiler trace: {profile_path}")
    profiler.dump_stats(profile_path)
    write_result(result, result_path)
    print(profile_path)
    print(result_path)
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
