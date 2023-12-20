"""
Abstractions of protein conformations.
"""

__all__ = ["Conformation", "Discrete", "Continuous"]

from typing import Any, Optional
from jaxtyping import Float, Array
from equinox import AbstractVar

from ..core import Module, field
from ..typing import Real_, Int_

_MixtureWeights = Float[Array, "N"]


class Conformation(Module):
    """
    Base class for a protein conformation.

    Attributes
    ----------
    coordinate :
        The conformation at which to evaluate the model.
    """

    coordinate: AbstractVar[Any]
    # distribution: AbstractVar[Any]


class Discrete(Conformation):
    """
    A discrete-valued conformational coordinate.
    """

    coordinate: Int_ = field(default=0)
    # distribution: Optional[_MixtureWeights] = field(default=None)


class Continuous(Conformation):
    """
    A continuous conformational coordinate.
    """

    coordinate: Real_ = field(default=0.0)
