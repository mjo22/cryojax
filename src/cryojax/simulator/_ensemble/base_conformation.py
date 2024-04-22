"""
Representations of conformational variables.
"""

from typing import Any

from equinox import AbstractVar, Module


class AbstractConformation(Module, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: AbstractVar[Any]
