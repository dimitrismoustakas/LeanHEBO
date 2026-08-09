# SPDX-License-Identifier: MIT

"""CLI for small quality smoke runs; generated data are not performance claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.harness.results import write_result
from benchmarks.quality.objectives import OBJECTIVES, get_objective
from benchmarks.quality.runner import RunSettings, make_adapter, run_trial


def _comma_separated(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=("leanhebo", "upstream-hebo"),
        default="leanhebo",
    )
    parser.add_argument(
        "--cases",
        type=_comma_separated,
        default=list(OBJECTIVES),
        help="comma-separated toy objective names",
    )
    parser.add_argument("--seeds", type=_comma_separated, default=["0"])
    parser.add_argument("--evaluation-budget", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--random-samples", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--gp-initial-steps", type=int, default=2)
    parser.add_argument("--gp-update-steps", type=int, default=1)
    parser.add_argument("--posterior-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "quality",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = RunSettings(
        evaluation_budget=args.evaluation_budget,
        batch_size=args.batch_size,
        random_samples=args.random_samples,
        population_size=args.population_size,
        generations=args.generations,
        gp_initial_steps=args.gp_initial_steps,
        gp_update_steps=args.gp_update_steps,
        posterior_batch_size=args.posterior_batch_size,
        device=args.device,
        dtype=args.dtype,
    )
    failures = 0
    for case in args.cases:
        objective = get_objective(case)
        for seed_text in args.seeds:
            seed = int(seed_text)
            adapter = make_adapter(args.implementation, objective, settings, seed)
            result = run_trial(adapter, objective, seed)
            destination = args.output_directory / f"{result.run_id}.json"
            write_result(result, destination)
            print(destination)
            failures += int(bool(result.failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
