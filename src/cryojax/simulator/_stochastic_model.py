"""
Base class for stochastic models.
"""

__all__ = ["AbstractStochasticModel"]

from abc import abstractmethod

from typing import Any
from equinox import Module
from jaxtyping import PRNGKeyArray

from ..typing import Image


class AbstractStochasticModel(Module):
    """
    Base class for stochastic models.
    """

    @abstractmethod
    def sample(self, key: PRNGKeyArray, *args: Any) -> Image:
        """Sample a realization from the model."""
        raise NotImplementedError
