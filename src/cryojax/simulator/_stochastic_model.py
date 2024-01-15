"""
Base class for stochastic models.
"""

__all__ = ["StochasticModel"]

from abc import abstractmethod

from typing import Any
from equinox import Module, AbstractClassVar
from jaxtyping import PRNGKeyArray

from ..typing import Image


class StochasticModel(Module):
    """
    Base class for stochastic models.
    """

    is_real: AbstractClassVar[bool]

    @abstractmethod
    def sample(self, key: PRNGKeyArray, *args: Any) -> Image:
        """Sample a realization from the model."""
        raise NotImplementedError
