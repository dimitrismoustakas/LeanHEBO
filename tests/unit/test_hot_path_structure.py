# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_IMPORTS = {"numpy", "pandas", "pymoo"}


def _hot_modules() -> list[Path]:
    source_root = Path(__file__).resolve().parents[2] / "src" / "leanhebo"
    return [
        source_root / "optimizer.py",
        *sorted((source_root / "acquisition").glob("*.py")),
        *sorted((source_root / "gp").glob("*.py")),
        *sorted((source_root / "search").glob("*.py")),
    ]


def test_numerical_hot_path_has_no_numpy_pandas_or_pymoo_imports() -> None:
    violations: list[str] = []
    for module in _hot_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            imported: list[str]
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            else:
                continue
            for name in imported:
                if name.split(".", 1)[0] in _FORBIDDEN_IMPORTS:
                    violations.append(f"{module.name}:{node.lineno}: {name}")
    assert not violations, "forbidden hot-path imports:\n" + "\n".join(violations)
