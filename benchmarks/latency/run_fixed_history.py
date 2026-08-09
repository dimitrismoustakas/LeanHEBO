# SPDX-License-Identifier: MIT

"""Run bounded fixed-history suggestion-latency cells against one implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from benchmarks.harness.results import write_result
from benchmarks.latency.fixed_history import (
    SUPPORTED_BATCH_SIZES,
    SUPPORTED_DIMENSIONS,
    SUPPORTED_FAMILIES,
    SUPPORTED_OBSERVATIONS,
    Family,
    matrix_cells,
    run_fixed_history,
)
from benchmarks.quality.runner import RunSettings


def _strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _integers(value: str) -> list[int]:
    try:
        values = [int(item) for item in _strings(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    return values


def _positive_int(value: str) -> int:
    converted = int(value)
    if converted < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=("leanhebo", "upstream-hebo"),
        default="leanhebo",
    )
    parser.add_argument(
        "--families",
        type=_strings,
        default=["continuous"],
        help=f"comma-separated subset of {SUPPORTED_FAMILIES}",
    )
    parser.add_argument(
        "--dimensions",
        type=_integers,
        default=[5],
        help=f"comma-separated subset of {SUPPORTED_DIMENSIONS}",
    )
    parser.add_argument(
        "--observations",
        type=_integers,
        default=[16],
        help=f"comma-separated subset of {SUPPORTED_OBSERVATIONS}",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_integers,
        default=[1],
        help=f"comma-separated subset of {SUPPORTED_BATCH_SIZES}",
    )
    parser.add_argument("--seeds", type=_integers, default=[0])
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=1,
        help="fresh optimizer constructions per seed and cell; history and RNG seed stay fixed",
    )
    parser.add_argument("--random-samples", type=_positive_int, default=4)
    parser.add_argument("--population-size", type=_positive_int, default=100)
    parser.add_argument("--generations", type=int, default=99)
    parser.add_argument("--gp-initial-steps", type=int, default=100)
    parser.add_argument(
        "--gp-optimizer",
        choices=("psgld", "adam", "lbfgs"),
        default="psgld",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--torch-threads", type=_positive_int, default=1)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "fixed-history",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cells = matrix_cells(
        cast(list[Family], args.families),
        args.dimensions,
        args.observations,
        args.batch_sizes,
    )
    failures = 0
    for cell in cells:
        settings = RunSettings(
            evaluation_budget=cell.observations,
            batch_size=cell.batch_size,
            random_samples=args.random_samples,
            population_size=args.population_size,
            generations=args.generations,
            gp_initial_steps=args.gp_initial_steps,
            gp_update_steps=args.gp_initial_steps,
            posterior_batch_size=None,
            gp_optimizer=args.gp_optimizer,
            learning_rate=args.learning_rate,
            model_lifecycle="cold",
            torch_threads=args.torch_threads,
            device="cpu",
            dtype="float32",
        )
        for seed in args.seeds:
            result = run_fixed_history(
                args.implementation,
                cell,
                settings,
                seed,
                repeats=args.repeats,
            )
            destination = args.output_directory / f"{result.run_id}.json"
            write_result(result, destination)
            print(destination)
            failures += int(bool(result.failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
