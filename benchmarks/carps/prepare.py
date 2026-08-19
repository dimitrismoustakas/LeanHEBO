# SPDX-License-Identifier: MIT

"""Prepare the isolated CARP-S benchmark environment and data."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
WORKSPACE = HERE.parent / ".carps"
ENVIRONMENT = WORKSPACE / "venv"
TASK_DATA = WORKSPACE / "task_data"
YAHPO_DATA = TASK_DATA / "yahpo_data"

YAHPO_DATA_REPOSITORY = "https://github.com/slds-lmu/yahpo_data.git"
YAHPO_DATA_COMMIT = "efdab9072f63bd680396cd4b78b927c4a0caaad3"
YAHPO_SCENARIOS = ("lcbench", "rbv2_glmnet", "rbv2_rpart")

SMOKE = textwrap.dedent(
    """
    import math
    from importlib.metadata import version

    import numpy as np
    from carps.objective_functions.bbob import get_bbob_problem
    from carps.objective_functions.yahpo import YahpoObjectiveFunction
    from carps.utils.trials import TrialInfo
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    from leanhebo import LeanHEBO
    from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig
    from leanhebo.space import Float, Space
    import leanhebo_carps

    assert version("carps") == "1.1.0"
    assert version("HEBO") == "0.3.6"
    assert version("pip") == "26.2.1"
    assert np.__version__ == "1.26.4"

    _, bbob = get_bbob_problem(fid=1, instance=1, dimension=2, seed=0)
    assert math.isfinite(float(bbob([0.0, 0.0])))

    yahpo = YahpoObjectiveFunction("lcbench", "167184", "val_accuracy", seed=0)
    value = yahpo.evaluate(TrialInfo(config=yahpo.configspace.get_default_configuration(), seed=0))
    assert isinstance(value.cost, float) and math.isfinite(value.cost)

    lean = LeanHEBO(
        Space(Float("x", -1.0, 1.0)),
        config=LeanHEBOConfig(
            random_samples=2,
            runtime=RuntimeConfig(seed=0),
            gp=GPConfig(initial_steps=1, update_steps=1, kernel_initialization_samples=16),
            search=SearchConfig(population_size=4, generations=1, seed=1),
        ),
    )
    initial = lean.suggest(2)
    lean.observe(initial, np.asarray([float(row["x"]) ** 2 for row in initial.to_records()]))
    modeled = lean.suggest(1)
    lean.observe(modeled, np.asarray([float(modeled.to_records()[0]["x"]) ** 2]))

    upstream_space = DesignSpace().parse([{"name": "x", "type": "num", "lb": -1.0, "ub": 1.0}])
    upstream = HEBO(upstream_space, rand_sample=2, model_config={"num_epochs": 1}, scramble_seed=0)
    for _ in range(3):
        suggestion = upstream.suggest(1)
        upstream.observe(suggestion, np.square(suggestion[["x"]].to_numpy()))

    print("CARP-S, BBOB, YAHPO, LeanHEBO, and HEBO smoke checks passed")
    """
)


def run(
    arguments: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(arguments, cwd=cwd, env=env, check=True)


def output(arguments: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def environment_python() -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return ENVIRONMENT / directory / executable


def prepare_data(git: str) -> None:
    if YAHPO_DATA.exists() and not (YAHPO_DATA / ".git").exists():
        raise RuntimeError(f"YAHPO data path is not a Git checkout: {YAHPO_DATA}")
    if not YAHPO_DATA.exists():
        YAHPO_DATA.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                git,
                "clone",
                "--filter=blob:none",
                "--sparse",
                YAHPO_DATA_REPOSITORY,
                str(YAHPO_DATA),
            ]
        )

    origin = output([git, "remote", "get-url", "origin"], cwd=YAHPO_DATA)
    if origin != YAHPO_DATA_REPOSITORY:
        raise RuntimeError(f"unexpected YAHPO data origin: {origin}")
    if output([git, "status", "--porcelain"], cwd=YAHPO_DATA):
        raise RuntimeError(f"refusing to update a dirty YAHPO data checkout: {YAHPO_DATA}")

    run([git, "sparse-checkout", "init", "--cone"], cwd=YAHPO_DATA)
    run([git, "sparse-checkout", "set", *YAHPO_SCENARIOS], cwd=YAHPO_DATA)
    run([git, "fetch", "--depth", "1", "origin", YAHPO_DATA_COMMIT], cwd=YAHPO_DATA)
    run([git, "checkout", "--detach", "--quiet", YAHPO_DATA_COMMIT], cwd=YAHPO_DATA)
    if output([git, "rev-parse", "HEAD"], cwd=YAHPO_DATA) != YAHPO_DATA_COMMIT:
        raise RuntimeError("YAHPO data checkout did not resolve to the pinned commit")


def main() -> int:
    uv = shutil.which("uv")
    git = shutil.which("git")
    if uv is None or git is None:
        raise RuntimeError("preparing the CARP-S benchmark requires uv and Git on PATH")

    python = environment_python()
    if not python.exists():
        ENVIRONMENT.parent.mkdir(parents=True, exist_ok=True)
        run([uv, "venv", "--python", "3.11.9", str(ENVIRONMENT)])
    version_check = "import platform; print(platform.python_version())"
    if output([str(python), "-c", version_check]) != "3.11.9":
        raise RuntimeError(f"benchmark environment is not Python 3.11.9: {python}")

    run([uv, "pip", "sync", "--python", str(python), str(HERE / "requirements.lock.txt")])
    run([uv, "pip", "install", "--python", str(python), "--no-deps", "HEBO==0.3.6"])
    run([uv, "pip", "install", "--python", str(python), "--no-deps", "--editable", str(REPOSITORY)])
    run([uv, "pip", "install", "--python", str(python), "--no-deps", "--editable", str(HERE)])

    prepare_data(git)
    smoke_environment = os.environ.copy()
    smoke_environment["CARPS_TASK_DATA_DIR"] = str(TASK_DATA.resolve())
    run([str(python), "-c", SMOKE], cwd=REPOSITORY, env=smoke_environment)

    print(f"environment: {python}")
    print(f"CARPS_TASK_DATA_DIR: {TASK_DATA.resolve()}")
    print("NumPy 1.26.4 is intentional for the upstream HEBO/YAHPO benchmark environment;")
    print("LeanHEBO and the local CARP-S adapter are installed editable with --no-deps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
