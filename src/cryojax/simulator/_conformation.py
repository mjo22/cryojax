"""
Representations of conformational variables.
"""

from typing import Any
from equinox import AbstractVar, field

from equinox import Module

from ..typing import Integer
from ..core import error_if_negative


class AbstractConformation(Module, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: AbstractVar[Any]


class DiscreteConformation(AbstractConformation, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: Integer = field(converter=error_if_negative)
