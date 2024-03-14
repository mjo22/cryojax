"""
Methods for integrating the scattering potential onto the exit plane.
"""

from abc import abstractmethod

from equinox import Module

from .._potential import AbstractScatteringPotential
from .._config import ImageConfig

from ...typing import ComplexImage


class AbstractPotentialIntegrator(Module, strict=True):
    """Base class for a method of integrating the scattering
    potential onto the exit plane."""

    @abstractmethod
    def __call__(
        self, potential: AbstractScatteringPotential, config: ImageConfig
    ) -> ComplexImage:
        """Compute the scattering potential in the exit plane at
        the `ImageConfig` settings.

        **Arguments:**

        - `potential`: The scattering potential representation.

        - `config`: The configuration of the resulting image.
        """
        raise NotImplementedError
