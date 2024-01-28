"""
Representations of conformational variables.
"""

from abc import abstractmethod
from typing import Any
from equinox import AbstractVar, field
from typing_extensions import override

import jax.numpy as jnp
from equinox import Module

from ..typing import Int_


class AbstractConformation(Module):
    """
    A conformational variable wrapped in a Module.
    """

    _value: AbstractVar[Any]

    @abstractmethod
    def get(self) -> Any:
        return self._value


class DiscreteConformation(AbstractConformation):
    """
    A conformational variable wrapped in a Module.
    """

    _value: Int_ = field(converter=jnp.asarray)

    @override
    def get(self) -> Int_:
        return self._value
