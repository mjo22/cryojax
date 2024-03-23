"""
Methods for integrating the scattering potential onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

from equinox import Module

from ...typing import ComplexImage
from .._config import ImageConfig
from .._potential import AbstractScatteringPotential


ScatteringPotentialT = TypeVar(
    "ScatteringPotentialT", bound="AbstractScatteringPotential"
)


class AbstractPotentialIntegrator(Module, Generic[ScatteringPotentialT], strict=True):
    """Base class for a method of integrating the scattering
    potential onto the exit plane."""

    @abstractmethod
    def __call__(
        self, potential: ScatteringPotentialT, config: ImageConfig
    ) -> ComplexImage:
        """Compute the scattering potential in the exit plane at
        the `ImageConfig` settings.

        **Arguments:**

        - `potential`: The scattering potential representation.

        - `config`: The configuration of the resulting image.
        """
        raise NotImplementedError
