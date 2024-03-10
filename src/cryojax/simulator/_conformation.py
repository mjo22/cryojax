"""
Representations of conformational variables.
"""

from typing import Any
from equinox import AbstractVar, field

import jax.numpy as jnp
from equinox import Module

from ..typing import Int_


class AbstractConformation(Module, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: AbstractVar[Any]


class DiscreteConformation(AbstractConformation, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: Int_ = field(converter=jnp.asarray)
