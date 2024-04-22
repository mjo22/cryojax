"""
Representations of conformational variables.
"""

from typing import Any

from equinox import AbstractVar, Module


class AbstractConformation(Module, strict=True):
    """A conformational variable wrapped in a Module."""

    value: AbstractVar[Any]


AbstractConformation.__init__.__doc__ = """**Arguments:**

- `value`: The value of the integer conformation.
"""
