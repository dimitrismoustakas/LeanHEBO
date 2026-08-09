# SPDX-License-Identifier: MIT

"""Create an isolated environment for the audited upstream HEBO commit.

The helper deliberately installs nothing into LeanHEBO's environment. It uses ``uv`` to create a
separate virtual environment beneath ``benchmarks/.upstream`` and installs the exact Git checkout
there. The checkout and generated environment are ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_BENCHMARKS = _HERE.parent
_DEFAULT_ROOT = _BENCHMARKS / ".upstream"


@dataclass(frozen=True, slots=True)
class UpstreamManifest:
    name: str
    repository: str
    commit: str
    commit_date: str
    package_subdirectory: str
    import_name: str
    python: str
    requirements_sha256: str
    setup_sha256: str
    development_only: bool

    @classmethod
    def load(cls, path: Path) -> UpstreamManifest:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError(f"unsupported upstream manifest schema in {path}")
        manifest = cls(
            name=_required_string(raw, "name"),
            repository=_required_string(raw, "repository"),
            commit=_required_string(raw, "commit"),
            commit_date=_required_string(raw, "commit_date"),
            package_subdirectory=_required_string(raw, "package_subdirectory"),
            import_name=_required_string(raw, "import_name"),
            python=_required_string(raw, "python"),
            requirements_sha256=_required_string(raw, "requirements_sha256"),
            setup_sha256=_required_string(raw, "setup_sha256"),
            development_only=raw.get("development_only") is True,
        )
        if len(manifest.commit) != 40 or any(
            char not in "0123456789abcdef" for char in manifest.commit
        ):
            raise ValueError("upstream commit must be a full lowercase Git SHA")
        for name, digest in (
            ("requirements_sha256", manifest.requirements_sha256),
            ("setup_sha256", manifest.setup_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if Path(manifest.package_subdirectory).is_absolute():
            raise ValueError("package_subdirectory must be relative")
        if not manifest.development_only:
            raise ValueError("the upstream baseline must remain development-only")
        return manifest


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return value


def _run(arguments: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    return completed.stdout.strip()


def _git(checkout: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(checkout), *arguments])


def _ensure_checkout(manifest: UpstreamManifest, checkout: Path) -> None:
    checkout = checkout.resolve()
    if checkout.exists() and not checkout.is_dir():
        raise ValueError(f"checkout path is not a directory: {checkout}")
    if not checkout.exists():
        checkout.mkdir(parents=True)
    git_directory = checkout / ".git"
    if not git_directory.exists():
        if any(checkout.iterdir()):
            raise ValueError(f"refusing to initialize a non-empty checkout directory: {checkout}")
        _git(checkout, "init", "--quiet")
        _git(checkout, "remote", "add", "origin", manifest.repository)
    elif _git(checkout, "remote", "get-url", "origin") != manifest.repository:
        raise ValueError("existing checkout's origin does not match the pinned manifest")

    dirty = _git(checkout, "status", "--porcelain")
    if dirty:
        raise RuntimeError("refusing to update a dirty upstream benchmark checkout")
    try:
        head = _git(checkout, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        head = ""
    if head != manifest.commit:
        _git(checkout, "fetch", "--depth", "1", "origin", manifest.commit)
        _git(checkout, "checkout", "--detach", "--quiet", "FETCH_HEAD")
    actual = _git(checkout, "rev-parse", "HEAD")
    if actual != manifest.commit:
        raise RuntimeError(f"expected upstream {manifest.commit}, checked out {actual}")

    package = (checkout / manifest.package_subdirectory).resolve()
    if checkout not in package.parents:
        raise ValueError("package_subdirectory escapes the checkout")
    _verify_digest(package / "requirements.txt", manifest.requirements_sha256)
    _verify_digest(package / "setup.py", manifest.setup_sha256)


def _verify_digest(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"pinned file digest mismatch for {path}: {actual} != {expected}")


def _environment_python(environment: Path) -> Path:
    if platform.system() == "Windows":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _ensure_environment(
    manifest: UpstreamManifest,
    checkout: Path,
    environment: Path,
    *,
    python: str,
) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv was not found on PATH; install uv before preparing the baseline")
    environment = environment.resolve()
    interpreter = _environment_python(environment)
    if not interpreter.exists():
        environment.parent.mkdir(parents=True, exist_ok=True)
        _run([uv, "venv", str(environment), "--python", python])
    package = (checkout / manifest.package_subdirectory).resolve()
    _run([uv, "pip", "install", "--python", str(interpreter), str(package)])
    _run(
        [
            str(interpreter),
            "-c",
            (
                f"import {manifest.import_name}; "
                f"print(getattr({manifest.import_name}, '__version__', 'unknown'))"
            ),
        ]
    )
    return interpreter


def _write_receipt(
    destination: Path,
    manifest: UpstreamManifest,
    checkout: Path,
    interpreter: Path | None,
) -> None:
    receipt = {
        "schema_version": 1,
        "name": manifest.name,
        "repository": manifest.repository,
        "commit": manifest.commit,
        "checkout": str(checkout.resolve()),
        "python": None if interpreter is None else str(interpreter.resolve()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_HERE / "upstream-hebo.json",
        help="pinned upstream manifest",
    )
    parser.add_argument(
        "--checkout",
        type=Path,
        default=_DEFAULT_ROOT / "hebo-checkout",
        help="isolated Git checkout destination",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=_DEFAULT_ROOT / "hebo-venv",
        help="isolated virtual environment destination",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="uv Python request (defaults to the version in the manifest)",
    )
    parser.add_argument(
        "--checkout-only",
        action="store_true",
        help="fetch and validate the source without resolving its development dependencies",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = UpstreamManifest.load(args.manifest)
    _ensure_checkout(manifest, args.checkout)
    interpreter = None
    if not args.checkout_only:
        interpreter = _ensure_environment(
            manifest,
            args.checkout,
            args.environment,
            python=manifest.python if args.python is None else args.python,
        )
    receipt = _DEFAULT_ROOT / "receipt.json"
    _write_receipt(receipt, manifest, args.checkout, interpreter)
    print(f"validated upstream HEBO {manifest.commit}")
    print(f"checkout: {args.checkout.resolve()}")
    if interpreter is not None:
        print(f"python: {interpreter}")
    print(f"receipt: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
