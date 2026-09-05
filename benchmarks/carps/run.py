"""Run sequential CARP-S optimizers and save one JSONL file per task and seed."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


def run_one(task, optimizer_config, label: str, output: Path) -> bool:
    from carps.utils.loggingutils import CustomEncoder
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:

        def write(row):
            stream.write(json.dumps(row, cls=CustomEncoder) + "\n")
            stream.flush()

        write(
            {
                "optimizer": label,
                "task": task.name,
                "seed": task.seed,
                "n_trials": task.optimization_resources.n_trials,
                "metric": task.output_space.objectives[0],
                "optimum": task.objective_function.f_min,
                "settings": OmegaConf.to_container(optimizer_config, resolve=True),
            }
        )
        try:
            optimizer = instantiate(optimizer_config, _partial_=True)(task=task)
            optimizer.setup_optimizer()
            for trial in range(1, task.optimization_resources.n_trials + 1):
                start = time.perf_counter()
                suggestion = optimizer.ask()
                ask_seconds = time.perf_counter() - start
                value = task.objective_function.evaluate(suggestion)
                cost = float(value.cost)
                if value.status != 1 or not math.isfinite(cost):
                    raise ValueError(f"Objective failed at trial {trial}: {value}")
                start = time.perf_counter()
                try:
                    optimizer.tell(suggestion, value)
                finally:
                    # An evaluated point remains usable if tell fails.
                    write(
                        {
                            "trial": trial,
                            "config": dict(suggestion.config),
                            "cost": cost,
                            "ask_seconds": ask_seconds,
                            "tell_seconds": time.perf_counter() - start,
                        }
                    )
        except Exception as error:
            write({"error": f"{type(error).__name__}: {error}"})
            traceback.print_exc()
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=HERE / "development_tasks.json")
    parser.add_argument("--optimizer", default="leanhebo", help="optimizer name or YAML file")
    parser.add_argument("--label", help="name used in results, e.g. LeanHEBO-conditional")
    parser.add_argument(
        "--source", type=Path, default=REPOSITORY, help="LeanHEBO checkout to import"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--trials", type=int, help="override each task's budget for a smoke check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    # Select the checkout before importing any optimizer; each invocation uses one source.
    if not (args.source / "src" / "leanhebo" / "__init__.py").is_file():
        parser.error(f"Not a LeanHEBO checkout: {args.source}")
    sys.path.insert(0, str(args.source.resolve() / "src"))
    os.environ["CARPS_TASK_DATA_DIR"] = str(HERE.parent / ".carps" / "task_data")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"

    import torch
    from leanhebo_carps.tasks import make_task
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    config_path = Path(args.optimizer)
    if config_path.suffix != ".yaml":
        config_path = (
            HERE / "leanhebo_carps" / "configs" / "optimizer" / args.optimizer / "config.yaml"
        )
    config = OmegaConf.load(config_path)
    label = args.label or config.name
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    failures = 0
    for spec in tasks:
        budget = args.trials if args.trials is not None else spec["n_trials"]
        if budget < 1:
            raise ValueError("Trial budgets must be positive")
        for seed in args.seeds:
            task = make_task(spec["name"], budget, seed, spec.get("metric"))
            settings = OmegaConf.merge(config, {"seed": seed}).optimizer
            output = args.output / label / task.name / str(seed) / "run.jsonl"
            completed = run_one(task, settings, label, output)
            failures += not completed
            print(
                f"{label} {task.name} seed={seed}: {'completed' if completed else 'failed'}",
                flush=True,
            )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
