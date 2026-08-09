# SPDX-License-Identifier: MIT

"""Typed LeanHEBO errors."""


class LeanHEBOError(RuntimeError):
    """Base class for runtime failures raised by LeanHEBO."""


class NumericalError(LeanHEBOError):
    """Raised when GP fitting or posterior evaluation remains numerically invalid."""


class CheckpointError(LeanHEBOError):
    """Raised when checkpoint state is malformed or incompatible."""


class SpaceMismatchError(LeanHEBOError, ValueError):
    """Raised when an encoded batch belongs to another compiled design space."""
