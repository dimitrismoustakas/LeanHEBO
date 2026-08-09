# SPDX-License-Identifier: MIT

"""Tensor-native batch and observation data APIs."""

from typing import TYPE_CHECKING, Any

from leanhebo.data.batch import CandidateBatch, EncodedBatch

if TYPE_CHECKING:
    from leanhebo.data.store import NonFinitePolicy, ObservationBatch, ObservationStore

__all__ = [
    "CandidateBatch",
    "EncodedBatch",
    "NonFinitePolicy",
    "ObservationBatch",
    "ObservationStore",
]


def __getattr__(name: str) -> Any:
    # CompiledSpace imports the adapter package.  Keeping store exports lazy
    # avoids a package-initialization cycle while preserving the concise public
    # ``from leanhebo.data import ObservationStore`` spelling.
    if name in {"NonFinitePolicy", "ObservationBatch", "ObservationStore"}:
        from leanhebo.data import store

        return getattr(store, name)
    raise AttributeError(name)
