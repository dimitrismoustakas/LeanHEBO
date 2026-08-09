# SPDX-License-Identifier: MIT

"""Chunked append-only observation state and exact duplicate membership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
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
    transformed_y: torch.Tensor | None
    observation_version: int
    transform_version: int

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
        retain_decoded: bool = False,
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
        self.retain_decoded = retain_decoded
        self._x_chunks: list[EncodedBatch] = []
        self._y_chunks: list[torch.Tensor] = []
        self._decoded_chunks: list[tuple[dict[str, object], ...]] = []
        self._keys = CanonicalKeySet(space)
        self._cache: ObservationBatch | None = None
        self._transformed_y: torch.Tensor | None = None
        self._transformed_observation_version: int | None = None
        self.observation_version = 0
        self.transform_version = 0
        self.discarded_count = 0
        self._lock = RLock()

    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self._x_chunks)

    @property
    def raw_data_version(self) -> int:
        return self.observation_version

    @property
    def chunk_count(self) -> int:
        return len(self._x_chunks)

    @property
    def encoded_chunks(self) -> tuple[EncodedBatch, ...]:
        with self._lock:
            return tuple(chunk.clone() for chunk in self._x_chunks)

    @property
    def y_chunks(self) -> tuple[torch.Tensor, ...]:
        with self._lock:
            return tuple(chunk.clone() for chunk in self._y_chunks)

    @property
    def records(self) -> list[dict[str, object]] | None:
        if not self.retain_decoded:
            return None
        return [record for chunk in self._decoded_chunks for record in chunk]

    @property
    def transform_is_stale(self) -> bool:
        return self._transformed_observation_version != self.observation_version

    def append(
        self,
        x: CandidateBatch | EncodedBatch | object,
        y: torch.Tensor | np.ndarray | Sequence[float] | float,
    ) -> int:
        """Append one valid chunk and return its retained row count."""

        source_candidates: CandidateBatch | None = None
        if isinstance(x, CandidateBatch):
            self.space._validate_fingerprint(x.space_fingerprint)
            self.space.validate_encoded(x.encoded)
            encoded = x.encoded  # direct path: no adapter or parameter codec call
            source_candidates = x
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
            if source_candidates is not None:
                source_candidates = source_candidates.select(finite)
        if len(encoded) == 0:
            return 0
        if self.retain_decoded:
            if source_candidates is None or source_candidates.decoded_columns is None:
                source_candidates = self.space.decode(encoded)
            records = tuple(source_candidates.to_records())
        else:
            records = ()
        with self._lock:
            self._x_chunks.append(encoded)
            self._y_chunks.append(outcomes.detach().clone())
            if self.retain_decoded:
                self._decoded_chunks.append(records)
            self._keys.add(encoded)
            self.observation_version += 1
            self._transformed_y = None
            self._transformed_observation_version = None
            self._cache = None
        return len(encoded)

    observe = append

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

    def contains(self, value: CandidateBatch | EncodedBatch) -> torch.Tensor:
        if isinstance(value, CandidateBatch):
            self.space._validate_fingerprint(value.space_fingerprint)
            encoded = value.encoded
        else:
            encoded = value
        self.space.validate_encoded(encoded)
        with self._lock:
            return self._keys.contains(encoded)

    def add_keys(self, value: CandidateBatch | EncodedBatch) -> int:
        """Reserve candidate keys without appending objective data."""

        if isinstance(value, CandidateBatch):
            self.space._validate_fingerprint(value.space_fingerprint)
            encoded = value.encoded
        else:
            encoded = value
        self.space.validate_encoded(encoded)
        with self._lock:
            return self._keys.add(encoded)

    def key_snapshot(self) -> tuple[tuple[int, ...], ...]:
        """Return all observed and reserved canonical keys for checkpointing."""

        with self._lock:
            return self._keys.snapshot()

    def restore_keys(self, keys: object) -> int:
        """Restore validated observed or reserved keys from a checkpoint."""

        with self._lock:
            return self._keys.add_canonical(keys)

    def restore_versions(self, observation_version: object, transform_version: object) -> None:
        """Restore exact monotonic counters after rebuilding checkpoint chunks."""

        def validated(value: object, name: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"checkpoint {name} is invalid")
            return value

        restored_observation_version = validated(observation_version, "observation_version")
        restored_transform_version = validated(transform_version, "transform_version")
        with self._lock:
            self.observation_version = restored_observation_version
            self.transform_version = restored_transform_version
            if self._transformed_y is not None:
                self._transformed_observation_version = restored_observation_version
            self._cache = None

    def unique_mask(
        self, value: CandidateBatch | EncodedBatch, *, include_pending: bool = True
    ) -> torch.Tensor:
        if isinstance(value, CandidateBatch):
            self.space._validate_fingerprint(value.space_fingerprint)
            encoded = value.encoded
        else:
            encoded = value
        self.space.validate_encoded(encoded)
        with self._lock:
            return self._keys.unique_mask(encoded, include_pending=include_pending)

    def _materialize_view(self) -> ObservationBatch:
        """Return the cached internal view; callers must not expose or mutate it."""

        with self._lock:
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
            self._cache = ObservationBatch(
                EncodedBatch(continuous, categorical),
                outcomes,
                self._transformed_y,
                self.observation_version,
                self.transform_version,
            )
            return self._cache

    def materialize(self) -> ObservationBatch:
        """Return a detached snapshot of the cached, concatenated observation state."""

        view = self._materialize_view()
        transformed = None if view.transformed_y is None else view.transformed_y.clone()
        return ObservationBatch(
            view.encoded.clone(),
            view.y.clone(),
            transformed,
            view.observation_version,
            view.transform_version,
        )

    @property
    def encoded(self) -> EncodedBatch:
        return self.materialize().encoded

    @property
    def continuous(self) -> torch.Tensor:
        return self.materialize().continuous

    @property
    def categorical(self) -> torch.Tensor:
        return self.materialize().categorical

    @property
    def y(self) -> torch.Tensor:
        return self.materialize().y

    @property
    def transformed_y(self) -> torch.Tensor | None:
        return self.materialize().transformed_y

    def set_transformed_y(
        self,
        values: torch.Tensor | np.ndarray | Sequence[float],
        *,
        observation_version: int | None = None,
    ) -> None:
        """Install transformed outcomes for the current raw-data version."""

        expected_version = self.observation_version
        if observation_version is not None and observation_version != expected_version:
            raise ValueError("transformed outcomes were computed from stale observations")
        transformed = self._coerce_outcomes(values, expected_rows=len(self))
        if not torch.isfinite(transformed).all():
            raise ValueError("transformed objective values must be finite")
        with self._lock:
            self._transformed_y = transformed.detach().clone()
            self._transformed_observation_version = expected_version
            self.transform_version += 1
            self._cache = None

    def clear(self) -> None:
        """Clear all data and reservations while retaining the compiled schema."""

        with self._lock:
            self._x_chunks.clear()
            self._y_chunks.clear()
            self._decoded_chunks.clear()
            self._keys.clear()
            self._cache = None
            self._transformed_y = None
            self._transformed_observation_version = None
            self.observation_version += 1
            self.transform_version += 1
            self.discarded_count = 0
