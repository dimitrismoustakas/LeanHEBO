# SPDX-License-Identifier: MIT

"""Explicit process-global Torch thread configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ProcessConfiguration:
    torch_num_threads: int
    torch_num_interop_threads: int


def configure_process(
    *,
    torch_num_threads: int | None = None,
    torch_num_interop_threads: int | None = None,
) -> ProcessConfiguration:
    """Set requested global Torch thread counts and return effective values.

    Call this helper before other Torch work. Torch only permits its inter-op thread count
    to be changed before parallel work begins; if that point has passed, the original
    ``RuntimeError`` is re-raised with actionable context.
    """

    if torch_num_threads is not None:
        if torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")
        torch.set_num_threads(torch_num_threads)
    if torch_num_interop_threads is not None:
        if torch_num_interop_threads < 1:
            raise ValueError("torch_num_interop_threads must be positive")
        try:
            torch.set_num_interop_threads(torch_num_interop_threads)
        except RuntimeError as exc:
            raise RuntimeError(
                "Torch inter-op threads can only be configured before parallel work starts; "
                "call configure_process() at application startup"
            ) from exc
    return ProcessConfiguration(torch.get_num_threads(), torch.get_num_interop_threads())
