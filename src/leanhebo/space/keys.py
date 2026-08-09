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

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: CanonicalKey) -> bool:
        return key in self._keys

    def contains(self, value: EncodedBatch | CandidateBatch) -> torch.Tensor:
        encoded = self._encoded(value)
        keys = self.space.canonical_keys(encoded)
        return torch.tensor(
            [key in self._keys for key in keys], dtype=torch.bool, device=encoded.device
        )

    def add(self, value: EncodedBatch | CandidateBatch) -> int:
        previous = len(self._keys)
        self._keys.update(self.space.canonical_keys(value))
        return len(self._keys) - previous

    add_keys = add

    def unique_mask(
        self, value: EncodedBatch | CandidateBatch, *, include_pending: bool = True
    ) -> torch.Tensor:
        """Mark unseen rows, optionally treating earlier query rows as pending."""

        encoded = self._encoded(value)
        pending: set[CanonicalKey] = set()
        result: list[bool] = []
        for key in self.space.canonical_keys(encoded):
            unseen = key not in self._keys and (not include_pending or key not in pending)
            result.append(unseen)
            if include_pending:
                pending.add(key)
        return torch.tensor(result, dtype=torch.bool, device=encoded.device)

    def clear(self) -> None:
        self._keys.clear()

    def snapshot(self) -> tuple[CanonicalKey, ...]:
        """Return a stable primitive snapshot suitable for checkpointing."""

        return tuple(sorted(self._keys))

    def add_canonical(self, keys: object) -> int:
        """Restore validated canonical keys without reconstructing candidate tensors."""

        if not isinstance(keys, (list, tuple)):
            raise TypeError("canonical key state must be a sequence")
        validated: set[CanonicalKey] = set()
        for key in keys:
            if not isinstance(key, (list, tuple)) or len(key) != len(self.space):
                raise ValueError("canonical key has an incompatible design-space width")
            if any(
                isinstance(component, bool) or not isinstance(component, int) for component in key
            ):
                raise TypeError("canonical key components must be integers")
            validated.add(tuple(key))
        previous = len(self._keys)
        self._keys.update(validated)
        return len(self._keys) - previous

    def _encoded(self, value: EncodedBatch | CandidateBatch) -> EncodedBatch:
        if isinstance(value, CandidateBatch):
            self.space._validate_fingerprint(value.space_fingerprint)
            encoded = value.encoded
        else:
            encoded = value
        self.space.validate_encoded(encoded)
        return encoded
