"""
Abstractions of protein conformations.
"""

__all__ = ["Conformation", "Discrete", "Continuous"]

from abc import ABCMeta
from typing import Any

from ..core import Module, field
from ..types import Real_, Integer_


class Conformation(Module, metaclass=ABCMeta):
    """
    Base class for a protein conformation.

    Attributes
    ----------
    coordinate : The conformation at which to evaluate the model.
    """

    coordinate: Any = field()


class Discrete(Conformation):
    """
    A discrete-valued conformational coordinate.
    """

    coordinate: Integer_ = field(default=0)


class Continuous(Conformation):
    """
    A continuous conformational coordinate.
    """

    coordinate: Real_ = field(default=0.0)
