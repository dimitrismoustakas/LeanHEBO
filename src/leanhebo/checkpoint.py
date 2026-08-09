# SPDX-License-Identifier: MIT

"""Versioned, tensor-and-primitive-only LeanHEBO checkpoints."""

from __future__ import annotations

import os
import pickle
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from leanhebo.errors import CheckpointError

CHECKPOINT_KIND = "leanhebo.optimizer"
CHECKPOINT_SCHEMA_VERSION = 1


def make_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": CHECKPOINT_KIND,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "payload": dict(payload),
    }


def save_checkpoint(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Atomically save a versioned checkpoint using Torch's tensor-aware format."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(make_checkpoint(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load and validate a checkpoint without permitting arbitrary pickled objects."""

    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
        struct.error,
    ) as exc:
        raise CheckpointError(f"failed to load LeanHEBO checkpoint: {exc}") from exc
    if not isinstance(state, Mapping) or state.get("kind") != CHECKPOINT_KIND:
        raise CheckpointError("file is not a LeanHEBO optimizer checkpoint")
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema: {state.get('schema_version')!r}")
    payload = state.get("payload")
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload is missing or malformed")
    return dict(payload)
