# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from leanhebo import LeanHEBO
from leanhebo.checkpoint import load_checkpoint, save_checkpoint
from leanhebo.errors import CheckpointError


def test_checkpoint_round_trip_uses_primitive_tensor_payload(tmp_path: object) -> None:
    path = tmp_path / "run.leanhebo"  # type: ignore[operator]
    payload = {"count": 3, "nested": {"values": torch.arange(4)}}
    save_checkpoint(path, payload)
    restored = load_checkpoint(path, map_location="cpu")
    assert restored["count"] == 3
    torch.testing.assert_close(restored["nested"]["values"], torch.arange(4))


@pytest.mark.parametrize("contents", [b"", b"junk", b"PK"])
def test_corrupt_checkpoint_always_raises_typed_error(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "corrupt.leanhebo"
    path.write_bytes(contents)

    with pytest.raises(CheckpointError, match="failed to load"):
        load_checkpoint(path)


def test_safe_but_malformed_optimizer_payload_raises_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "malformed.leanhebo"
    save_checkpoint(path, {"config": {}})

    with pytest.raises(CheckpointError, match="payload is malformed"):
        LeanHEBO.load(path, map_location="cpu")
