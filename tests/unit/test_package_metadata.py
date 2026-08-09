# SPDX-License-Identifier: MIT

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import cast

import leanhebo


def test_runtime_version_matches_distribution_metadata() -> None:
    project_root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast(dict[str, object], metadata["project"])

    assert leanhebo.__version__ == project["version"] == version("leanhebo")
