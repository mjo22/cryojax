"""
Abstractions of protein conformations.
"""

__all__ = ["Conformation", "Discrete", "Continuous"]

from abc import ABCMeta, abstractmethod
from typing import Any

import jax.numpy as jnp

from ..core import Module, field, Real_, Integer_


class Conformation(Module, metaclass=ABCMeta):
    """
    Base class for a protein conformation.
    """

    @property
    @abstractmethod
    def coordinate(self) -> Any:
        raise NotImplementedError


class Discrete(Conformation):
    """
    A discrete-valued conformational coordinate.

    Attributes
    ----------
    m : The conformation at which to evaluate the model.
    """

    m: Integer_ = field(default=0)

    @property
    def coordinate(self) -> Integer_:
        return self.m


class Continuous(Conformation):
    """
    A continuous conformational coordinate.

    Attributes
    ----------
    z : The conformation at which to evaluate the model.
    """

    z: Real_ = field(default=0.0)

    @property
    def coordinate(self) -> Real_:
        return self.z
