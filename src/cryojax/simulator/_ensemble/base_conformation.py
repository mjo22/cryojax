"""
Representations of conformational variables.
"""

from typing import Any

from equinox import AbstractVar, Module


class AbstractConformationalVariable(Module, strict=True):
    """A conformational variable wrapped in an `equinox.Module`."""

    value: AbstractVar[Any]


AbstractConformationalVariable.__init__.__doc__ = """**Arguments:**

- `value`: The value of the integer conformation.
"""
