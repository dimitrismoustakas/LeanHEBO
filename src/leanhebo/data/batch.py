# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Tensor-native batch containers used at public and internal boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import torch


def _validate_tensors(continuous: torch.Tensor, categorical: torch.Tensor) -> None:
    if not isinstance(continuous, torch.Tensor) or not isinstance(categorical, torch.Tensor):
        raise TypeError("continuous and categorical must be torch.Tensor instances")
    if continuous.ndim != 2 or categorical.ndim != 2:
        raise ValueError("encoded tensors must both have shape [rows, columns]")
    if continuous.shape[0] != categorical.shape[0]:
        raise ValueError("continuous and categorical tensors must have the same row count")
    if not continuous.is_floating_point():
        raise TypeError("the continuous tensor must use a floating dtype")
    if categorical.dtype != torch.int64:
        raise TypeError("the categorical tensor must use torch.int64")
    if continuous.device != categorical.device:
        raise ValueError("encoded tensors must be on the same device")


def _normalize_rows(
    rows: torch.Tensor | slice | Sequence[int],
    *,
    count: int,
    device: torch.device,
) -> torch.Tensor | slice:
    if isinstance(rows, slice):
        return rows
    index = torch.as_tensor(rows, device=device)
    if index.dtype == torch.bool:
        if index.ndim != 1 or index.numel() != count:
            raise ValueError("a Boolean row mask must have one entry per batch row")
        return index
    return index.to(torch.int64).reshape(-1)


@dataclass(frozen=True, slots=True, eq=False)
class EncodedBatch:
    """The native mixed-space representation.

    Numeric optimization coordinates and categorical codes remain separate so
    downstream models never need to recover integer codes from a floating
    concatenation.
    """

    continuous: torch.Tensor
    categorical: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensors(self.continuous, self.categorical)

    def __len__(self) -> int:
        return self.continuous.shape[0]

    @property
    def device(self) -> torch.device:
        return self.continuous.device

    @property
    def dtype(self) -> torch.dtype:
        return self.continuous.dtype

    @property
    def n_continuous(self) -> int:
        return self.continuous.shape[1]

    @property
    def n_categorical(self) -> int:
        return self.categorical.shape[1]

    def to(
        self,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> EncodedBatch:
        """Move the batch without ever casting categorical codes to float."""

        target_dtype = self.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise TypeError("EncodedBatch continuous dtype must be floating point")
        return EncodedBatch(
            self.continuous.to(device=device, dtype=target_dtype, non_blocking=non_blocking),
            self.categorical.to(device=device, dtype=torch.int64, non_blocking=non_blocking),
        )

    def detach(self) -> EncodedBatch:
        return EncodedBatch(self.continuous.detach(), self.categorical.detach())

    def clone(self) -> EncodedBatch:
        return EncodedBatch(self.continuous.clone(), self.categorical.clone())

    def select(self, rows: torch.Tensor | slice | Sequence[int]) -> EncodedBatch:
        """Select rows while retaining the two-dimensional batch shape."""

        index = _normalize_rows(rows, count=len(self), device=self.device)
        if isinstance(index, slice) or index.dtype == torch.bool:
            continuous = self.continuous[index]
            categorical = self.categorical[index]
        else:
            continuous = self.continuous.index_select(0, index)
            categorical = self.categorical.index_select(0, index)
        return EncodedBatch(continuous, categorical)

    def to_dense(self) -> torch.Tensor:
        """Return continuous coordinates followed by categorical codes."""

        if self.n_categorical == 0:
            return self.continuous
        if self.n_continuous == 0:
            return self.categorical.to(dtype=self.dtype)
        return torch.cat((self.continuous, self.categorical.to(dtype=self.dtype)), dim=1)


@dataclass(frozen=True, slots=True, eq=False)
class CandidateBatch:
    """Candidates with both encoded tensors and lazily materialized user values."""

    continuous: torch.Tensor
    categorical: torch.Tensor
    space_fingerprint: str
    decoded_columns: Mapping[str, Sequence[object]] | None = None
    activity: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_tensors(self.continuous, self.categorical)
        if not isinstance(self.space_fingerprint, str) or not self.space_fingerprint:
            raise ValueError("space_fingerprint must be a non-empty string")
        if self.activity is not None:
            if not isinstance(self.activity, torch.Tensor):
                raise TypeError("activity must be a torch.Tensor or None")
            if self.activity.dtype != torch.bool or self.activity.ndim != 2:
                raise TypeError("activity must be a rank-two Boolean tensor")
            if self.activity.shape[0] != len(self):
                raise ValueError("activity and encoded tensors must have the same row count")
            if self.activity.device != self.device:
                raise ValueError("activity and encoded tensors must share a device")
        if self.decoded_columns is not None:
            frozen_columns = {
                str(name): tuple(values) for name, values in self.decoded_columns.items()
            }
            invalid = [name for name, values in frozen_columns.items() if len(values) != len(self)]
            if invalid:
                raise ValueError(
                    "decoded column lengths differ from the encoded row count: "
                    + ", ".join(invalid)
                )
            object.__setattr__(self, "decoded_columns", MappingProxyType(frozen_columns))

    def __len__(self) -> int:
        return self.continuous.shape[0]

    @property
    def encoded(self) -> EncodedBatch:
        """Expose the original encoded tensors without re-encoding user values."""

        return EncodedBatch(self.continuous, self.categorical)

    @property
    def device(self) -> torch.device:
        return self.continuous.device

    @property
    def dtype(self) -> torch.dtype:
        return self.continuous.dtype

    def to(
        self,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> CandidateBatch:
        """Move encoded state while retaining already decoded boundary values."""

        encoded = self.encoded.to(device, dtype=dtype, non_blocking=non_blocking)
        return CandidateBatch(
            encoded.continuous,
            encoded.categorical,
            self.space_fingerprint,
            self.decoded_columns,
            None if self.activity is None else self.activity.to(device=encoded.device),
        )

    def select(self, rows: torch.Tensor | slice | Sequence[int]) -> CandidateBatch:
        index = _normalize_rows(rows, count=len(self), device=self.device)
        if isinstance(index, slice) or index.dtype == torch.bool:
            continuous = self.continuous[index]
            categorical = self.categorical[index]
            activity = None if self.activity is None else self.activity[index]
        else:
            continuous = self.continuous.index_select(0, index)
            categorical = self.categorical.index_select(0, index)
            activity = None if self.activity is None else self.activity.index_select(0, index)
        columns = None
        if self.decoded_columns is not None:
            if isinstance(index, slice):
                indices = tuple(range(len(self)))[index]
            elif index.dtype == torch.bool:
                indices = tuple(index.nonzero(as_tuple=False).flatten().cpu().tolist())
            else:
                indices = tuple(index.cpu().tolist())
            columns = {
                name: tuple(values[index] for index in indices)
                for name, values in self.decoded_columns.items()
            }
        return CandidateBatch(
            continuous,
            categorical,
            self.space_fingerprint,
            columns,
            activity,
        )

    def to_dense(self) -> torch.Tensor:
        return self.encoded.to_dense()

    def to_records(self) -> list[dict[str, object]]:
        """Convert decoded values to row dictionaries in schema order."""

        if self.decoded_columns is None:
            raise ValueError("this CandidateBatch has no decoded column metadata")
        names = tuple(self.decoded_columns)
        if self.activity is not None:
            if self.activity.shape[1] != len(names):
                raise ValueError("activity columns do not match decoded column metadata")
            active = self.activity.detach().to(device="cpu")
            return [
                {
                    name: self.decoded_columns[name][row]
                    for column, name in enumerate(names)
                    if bool(active[row, column])
                }
                for row in range(len(self))
            ]
        return [
            {name: self.decoded_columns[name][row] for name in names} for row in range(len(self))
        ]

    def to_numpy(self) -> np.ndarray:
        """Return an object array in the design space's public column order."""

        if self.decoded_columns is None:
            raise ValueError("this CandidateBatch has no decoded column metadata")
        names = tuple(self.decoded_columns)
        result = np.empty((len(self), len(names)), dtype=object)
        for column, name in enumerate(names):
            result[:, column] = self.decoded_columns[name]
        return result
