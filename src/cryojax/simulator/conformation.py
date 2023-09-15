"""
Abstractions of protein conformations.
"""

__all__ = ["Conformation", "Discrete", "Continuous"]

from abc import ABCMeta

from ..core import CryojaxObject, dataclass, field, Parameter


@dataclass
class Conformation(CryojaxObject, metaclass=ABCMeta):
    """
    Base class for a protein conformation.
    """


@dataclass
class Discrete(Conformation):
    """
    A discrete-valued conformational coordinate.

    Attributes
    ----------
    m : `int`
        The conformation at which to evaluate the model.
    M : `int`
        The number of conformations in the model.
    """

    m: int = field(default=0)
    M: int = field(pytree_node=False, default=1)


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
