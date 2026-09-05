# SPDX-License-Identifier: MIT

"""Prepare the isolated CARP-S benchmark environment and data."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
WORKSPACE = HERE.parent / ".carps"
ENVIRONMENT = WORKSPACE / "venv"
TASK_DATA = WORKSPACE / "task_data"
YAHPO_DATA = TASK_DATA / "yahpo_data"

YAHPO_DATA_REPOSITORY = "https://github.com/slds-lmu/yahpo_data.git"
YAHPO_DATA_COMMIT = "efdab9072f63bd680396cd4b78b927c4a0caaad3"
YAHPO_SCENARIOS = (
    "lcbench",
    "rbv2_aknn",
    "rbv2_glmnet",
    "rbv2_ranger",
    "rbv2_rpart",
    "rbv2_svm",
    "rbv2_xgboost",
)


def run(arguments: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def environment_python() -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return ENVIRONMENT / directory / executable


def prepare_data() -> None:
    if not YAHPO_DATA.exists():
        YAHPO_DATA.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                YAHPO_DATA_REPOSITORY,
                str(YAHPO_DATA),
            ]
        )

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=YAHPO_DATA, check=True, capture_output=True
    )
    if status.stdout.strip():
        raise RuntimeError(f"refusing to update a dirty YAHPO data checkout: {YAHPO_DATA}")

    run(["git", "sparse-checkout", "set", *YAHPO_SCENARIOS], cwd=YAHPO_DATA)
    run(["git", "fetch", "--depth", "1", "origin", YAHPO_DATA_COMMIT], cwd=YAHPO_DATA)
    run(["git", "checkout", "--detach", "--quiet", YAHPO_DATA_COMMIT], cwd=YAHPO_DATA)


def main() -> int:
    python = environment_python()
    if not python.exists():
        ENVIRONMENT.parent.mkdir(parents=True, exist_ok=True)
        run(["uv", "venv", "--python", "3.11.9", str(ENVIRONMENT)])

    run(["uv", "pip", "sync", "--python", str(python), str(HERE / "requirements.lock.txt")])
    install = ["uv", "pip", "install", "--python", str(python), "--no-deps"]
    run([*install, "HEBO==0.3.6"])
    for source in (REPOSITORY, HERE):
        run([*install, "--editable", str(source)])

    prepare_data()
    print(f"environment: {python}")
    print(f"CARPS_TASK_DATA_DIR: {TASK_DATA.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
