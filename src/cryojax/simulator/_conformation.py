"""
Representations of conformational variables.
"""

from typing import Any
from jaxtyping import Shaped
from equinox import Module, AbstractVar, field

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

    value: Shaped[Integer, "..."] = field(converter=error_if_negative)
