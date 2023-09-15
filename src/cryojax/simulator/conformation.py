"""
Abstractions of protein conformations.
"""

__all__ = ["Conformation", "Discrete", "Continuous"]

from abc import ABCMeta, abstractmethod
from typing import Any

from ..core import CryojaxObject, dataclass, field, Parameter


@dataclass
class Conformation(CryojaxObject, metaclass=ABCMeta):
    """
    Base class for a protein conformation.
    """

    @property
    @abstractmethod
    def coordinate(self) -> Any:
        raise NotImplementedError


@dataclass
class Discrete(Conformation):
    """
    A discrete-valued conformational coordinate.

    Attributes
    ----------
    m : `int`
        The conformation at which to evaluate the model.
    """

    m: int = field(default=0)

    @property
    def coordinate(self) -> int:
        return self.m


@dataclass
class Continuous(Conformation):
    """
    A continuous conformational coordinate.

    Attributes
    ----------
    z : `cryojax.core.Parameter`
        The conformation at which to evaluate the model.
    """

    z: Parameter = field(default=0.0)

    @property
    def coordinate(self) -> Parameter:
        return self.z
