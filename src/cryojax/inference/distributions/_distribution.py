"""
Base class for a cryojax distribution.
"""

from abc import abstractmethod
from typing import Any

from jaxtyping import PRNGKeyArray
from equinox import Module, AbstractVar

from ...simulator import AbstractPipeline
from ...typing import Real_, Image


class AbstractDistribution(Module, strict=True):
    """An image formation model equipped with a probabilistic model."""

    pipeline: AbstractVar[AbstractPipeline]

    @abstractmethod
    def log_likelihood(self, observed: Image) -> Real_:
        """Evaluate the log likelihood.

        **Arguments:**

        - `observed` : The observed data in real or fourier space.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, key: PRNGKeyArray, *, get_real: bool = True) -> Image:
        """Sample from the distribution.

        **Arguments:**

        - `key` : The RNG key or key(s). See `AbstractPipeline.sample` for
                  more documentation.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, *, get_real: bool = True) -> Image:
        """Render the image formation model."""
        raise NotImplementedError
