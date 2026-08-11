# SPDX-License-Identifier: MIT

"""Exact canonical-key membership for compiled mixed spaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from leanhebo.data.batch import CandidateBatch, EncodedBatch
from leanhebo.space.compiled import CompiledSpace

CanonicalKey = tuple[int, ...]


@dataclass(slots=True)
class CanonicalKeySet:
    """A collision-free Python set over exact compiled-space key tuples."""

    space: CompiledSpace
    _keys: set[CanonicalKey] = field(default_factory=set, init=False, repr=False)

    def add(self, value: EncodedBatch | CandidateBatch) -> int:
        previous = len(self._keys)
        self._keys.update(self.space.canonical_keys(value))
        return len(self._keys) - previous

    def unique_mask(self, value: EncodedBatch | CandidateBatch) -> torch.Tensor:
        """Mark rows absent from history and earlier positions in this batch."""

        encoded = self._encoded(value)
        seen: set[CanonicalKey] = set()
        result: list[bool] = []
        for key in self.space.canonical_keys(encoded):
            unseen = key not in self._keys and key not in seen
            result.append(unseen)
            seen.add(key)
        return torch.tensor(result, dtype=torch.bool, device=encoded.device)

    def clear(self) -> None:
        self._keys.clear()

    def snapshot(self) -> tuple[CanonicalKey, ...]:
        """Return a stable primitive snapshot suitable for checkpointing."""

        return tuple(sorted(self._keys))

    def _encoded(self, value: EncodedBatch | CandidateBatch) -> EncodedBatch:
        if isinstance(value, CandidateBatch):
            self.space._validate_fingerprint(value.space_fingerprint)
            encoded = value.encoded
        else:
            encoded = value
        self.space.validate_encoded(encoded)
        return encoded
