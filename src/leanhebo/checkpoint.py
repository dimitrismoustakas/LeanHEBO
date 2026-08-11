# SPDX-License-Identifier: MIT

"""Safe, atomic persistence for LeanHEBO optimizer state."""

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


def save_checkpoint(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Atomically save tensor-and-primitive optimizer state."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
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
    if not isinstance(state, Mapping):
        raise CheckpointError("checkpoint payload is malformed")
    return dict(state)
