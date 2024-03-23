"""
Representations of conformational variables.
"""

from typing import Any

from equinox import AbstractVar, field, Module
from jaxtyping import Shaped

from ..core import error_if_negative
from ..typing import Integer


class AbstractConformation(Module, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: AbstractVar[Any]


class DiscreteConformation(AbstractConformation, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: Shaped[Integer, "..."] = field(converter=error_if_negative)
