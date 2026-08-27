# SPDX-License-Identifier: MIT

"""Chunked append-only observation state and exact duplicate membership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch

from leanhebo.data.batch import CandidateBatch, EncodedBatch
from leanhebo.space.compiled import CompiledSpace, _dtype_from_value
from leanhebo.space.keys import CanonicalKeySet
from leanhebo.space.space import Space

NonFinitePolicy: TypeAlias = Literal["raise", "drop"]


@dataclass(frozen=True, slots=True, eq=False)
class ObservationBatch:
    """A cached materialized view of chunked observation state."""

    encoded: EncodedBatch
    y: torch.Tensor

    def __len__(self) -> int:
        return len(self.encoded)

    @property
    def continuous(self) -> torch.Tensor:
        return self.encoded.continuous

    @property
    def categorical(self) -> torch.Tensor:
        return self.encoded.categorical


class ObservationStore:
    """Append observations without repeated full-history concatenation.

    The default ``nonfinite='drop'`` policy discards rows with NaN or infinite
    outcomes and records their count.  ``nonfinite='raise'`` is available for
    strict objective pipelines.  Materialization is cached until another chunk
    is appended.
    """

    def __init__(
        self,
        space: CompiledSpace | Space,
        *,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str = "cpu",
        nonfinite: NonFinitePolicy = "drop",
    ) -> None:
        if isinstance(space, Space):
            space = space.compile(dtype=torch.float32 if dtype is None else dtype)
        elif dtype is not None:
            requested = _dtype_from_value(dtype)
            if requested != space.dtype:
                raise ValueError("dtype must match the already compiled space")
        if nonfinite not in ("raise", "drop"):
            raise ValueError("nonfinite policy must be 'raise' or 'drop'")
        self.space = space
        self.device = torch.device(device)
        self.dtype = space.dtype
        self.nonfinite = nonfinite
        self._x_chunks: list[EncodedBatch] = []
        self._y_chunks: list[torch.Tensor] = []
        self._keys = CanonicalKeySet(space)
        self._cache: ObservationBatch | None = None
        self.observation_version = 0
        self.discarded_count = 0

    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self._x_chunks)

    def append(
        self,
        x: CandidateBatch | EncodedBatch | object,
        y: torch.Tensor | np.ndarray | Sequence[float] | float,
    ) -> int:
        """Append one valid chunk and return its retained row count."""

        if isinstance(x, CandidateBatch):
            self.space._validate_fingerprint(x.space_fingerprint)
            self.space.validate_encoded(x.encoded)
            encoded = x.encoded  # direct path: no adapter or parameter codec call
        elif isinstance(x, EncodedBatch):
            self.space.validate_encoded(x)
            encoded = x
        else:
            encoded = self.space.encode(x)
        outcomes = self._coerce_outcomes(y, expected_rows=len(encoded))
        # Observation history is append-only. CandidateBatch avoids codec and adapter work,
        # but the store still owns a snapshot so later caller mutation cannot rewrite history
        # while leaving canonical keys and materialized caches inconsistent.
        encoded = encoded.to(self.device, dtype=self.dtype).detach().clone()
        finite = torch.isfinite(outcomes)
        invalid_count = int((~finite).sum().item())
        if invalid_count and self.nonfinite == "raise":
            raise ValueError(f"objective contains {invalid_count} non-finite outcome(s)")
        if invalid_count:
            encoded = encoded.select(finite)
            outcomes = outcomes[finite]
            self.discarded_count += invalid_count
        if len(encoded) == 0:
            return 0
        self._x_chunks.append(encoded)
        self._y_chunks.append(outcomes.detach().clone())
        self._keys.add(encoded)
        self.observation_version += 1
        self._cache = None
        return len(encoded)

    def _coerce_outcomes(
        self,
        values: torch.Tensor | np.ndarray | Sequence[float] | float,
        *,
        expected_rows: int,
    ) -> torch.Tensor:
        try:
            result = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        except (TypeError, ValueError) as error:
            raise TypeError("objective values must be numeric") from error
        if result.ndim == 0:
            result = result.reshape(1)
        elif result.ndim == 2 and result.shape[1] == 1:
            result = result[:, 0]
        elif result.ndim != 1:
            raise ValueError("single-objective outcomes must have shape [rows] or [rows, 1]")
        if result.shape[0] != expected_rows:
            raise ValueError(
                "candidate and objective batch lengths differ: "
                f"{expected_rows} != {result.shape[0]}"
            )
        return result

    def key_snapshot(self) -> tuple[tuple[int, ...], ...]:
        """Return the observed canonical keys."""

        return self._keys.snapshot()

    def unique_mask(self, value: CandidateBatch | EncodedBatch) -> torch.Tensor:
        if isinstance(value, CandidateBatch):
            self.space._validate_fingerprint(value.space_fingerprint)
            encoded = value.encoded
        else:
            encoded = value
        self.space.validate_encoded(encoded)
        return self._keys.unique_mask(encoded)

    def _materialize_view(self) -> ObservationBatch:
        """Return the cached internal view; callers must not expose or mutate it."""

        if self._cache is not None:
            return self._cache
        if self._x_chunks:
            continuous = torch.cat([chunk.continuous for chunk in self._x_chunks], dim=0)
            categorical = torch.cat([chunk.categorical for chunk in self._x_chunks], dim=0)
            outcomes = torch.cat(self._y_chunks, dim=0)
        else:
            continuous = torch.empty(
                (0, self.space.n_continuous), dtype=self.dtype, device=self.device
            )
            categorical = torch.empty(
                (0, self.space.n_categorical),
                dtype=torch.int64,
                device=self.device,
            )
            outcomes = torch.empty((0,), dtype=self.dtype, device=self.device)
        self._cache = ObservationBatch(EncodedBatch(continuous, categorical), outcomes)
        return self._cache

    def materialize(self) -> ObservationBatch:
        """Return a detached snapshot of the cached, concatenated observation state."""

        view = self._materialize_view()
        return ObservationBatch(view.encoded.clone(), view.y.clone())

    def clear(self) -> None:
        """Clear all data and reservations while retaining the compiled schema."""

        self._x_chunks.clear()
        self._y_chunks.clear()
        self._keys.clear()
        self._cache = None
        self.observation_version += 1
        self.discarded_count = 0
