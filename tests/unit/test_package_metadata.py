# SPDX-License-Identifier: MIT

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import cast

import leanhebo


def test_public_versions_match_distribution_metadata() -> None:
    project_root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast(dict[str, object], metadata["project"])
    citation_version = next(
        line.removeprefix("version:").strip().strip('"')
        for line in (project_root / "CITATION.cff").read_text(encoding="utf-8").splitlines()
        if line.startswith("version:")
    )

    assert leanhebo.__version__ == project["version"] == citation_version == version("leanhebo")
