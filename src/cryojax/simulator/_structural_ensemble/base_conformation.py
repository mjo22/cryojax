"""
Representations of conformational variables.
"""

from typing import Any

from equinox import AbstractVar, Module


class AbstractConformationalVariable(Module, strict=True):
    """A conformational variable wrapped in an `equinox.Module`."""

    value: AbstractVar[Any]
