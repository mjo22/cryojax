"""
Methods for integrating the scattering potential onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar
from typing_extensions import override

from equinox import Module
from jaxtyping import Array, Complex

from .._config import ImageConfig
from .._potential import AbstractSpecimenPotential, AbstractVoxelPotential


PotentialT = TypeVar("PotentialT", bound="AbstractSpecimenPotential")


class AbstractPotentialProjectionMethod(Module, Generic[PotentialT], strict=True):
    """Base class for a method of extracting projections of a potential."""

    @abstractmethod
    def compute_fourier_projected_potential(
        self,
        potential: PotentialT,
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


class AbstractVoxelPotentialProjectionMethod(
    AbstractPotentialProjectionMethod[AbstractVoxelPotential], strict=True
):
    """Base class for a method of extracting projections of a voxel-based potential."""

    @abstractmethod
    def compute_raw_fourier_projected_potential(
        self,
        potential: AbstractVoxelPotential,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError

    @override
    def compute_fourier_projected_potential(
        self,
        potential: AbstractVoxelPotential,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raw_fourier_projected_potential = self.compute_raw_fourier_projected_potential(
            potential, config
        )
        return config.rescale_to_pixel_size(
            potential.voxel_size * raw_fourier_projected_potential,
            potential.voxel_size,
            is_real=False,
        )
