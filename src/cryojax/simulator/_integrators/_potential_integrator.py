"""
Methods for integrating the scattering potential onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

from equinox import Module
from jaxtyping import Array, Complex

from ...typing import RealNumber
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
        self,
        potential: ScatteringPotentialT,
        wavelength_in_angstroms: RealNumber,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the scattering potential in the exit plane at
        the `ImageConfig` settings.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `wavelength_in_angstroms`: The wavelength of the electron beam.
        - `config`: The configuration of the resulting image.
        """
        raise NotImplementedError
