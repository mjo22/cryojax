"""
Methods for integrating the scattering potential onto the exit plane.
"""

from abc import abstractmethod
from typing import Any

from equinox import Module, AbstractVar

from .._potential import AbstractScatteringPotential
from .._config import ImageConfig

from ...typing import ComplexImage


class AbstractPotentialIntegrator(Module, strict=True):
    """Base class for a method of integrating the scattering
    potential onto the exit plane."""

    config: AbstractVar[ImageConfig]

    @abstractmethod
    def __call__(self, potential: AbstractScatteringPotential) -> ComplexImage:
        """Compute the scattering potential in the exit plane at
        the `ImageConfig` settings.

        **Arguments:**

        `potential`: The scattering potential representation.
        """
        raise NotImplementedError
