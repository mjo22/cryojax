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


class AbstractConformation(Module, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: AbstractVar[Any]

    @abstractmethod
    def get(self) -> Any:
        return self.value


class DiscreteConformation(AbstractConformation, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: Int_ = field(converter=jnp.asarray)

    @override
    def get(self) -> Int_:
        return self.value
