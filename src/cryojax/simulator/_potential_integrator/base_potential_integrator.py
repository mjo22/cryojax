"""
Methods for integrating the scattering potential directly onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar
from typing_extensions import override

from equinox import AbstractVar, Module
from jaxtyping import Array, Complex

from ...image import maybe_rescale_pixel_size
from .._instrument_config import InstrumentConfig
from .._potential_representation import AbstractVoxelPotential


PotentialT = TypeVar("PotentialT")
VoxelPotentialT = TypeVar("VoxelPotentialT", bound="AbstractVoxelPotential")


class AbstractPotentialIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a method of integrating a potential directly onto
    an imaging plane."""

    @abstractmethod
    def compute_fourier_integrated_potential(
        self,
        potential: PotentialT,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the scattering potential in the exit plane at
        the `InstrumentConfig` settings.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `wavelength_in_angstroms`: The wavelength of the electron beam.
        - `instrument_config`: The configuration of the resulting image.
        """
        raise NotImplementedError


class AbstractVoxelPotentialIntegrator(
    AbstractPotentialIntegrator[AbstractVoxelPotential],
    Generic[VoxelPotentialT],
    strict=True,
):
    """Base class for a method of integrating a voxel-based potential."""

    pixel_rescaling_method: AbstractVar[str]

    @abstractmethod
    def compute_raw_fourier_image(
        self,
        potential: VoxelPotentialT,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError

    @override
    def compute_fourier_integrated_potential(
        self,
        potential: AbstractVoxelPotential,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        fourier_projected_potential_without_postprocess = self.compute_raw_fourier_image(
            potential,  # type: ignore
            instrument_config,
        )
        return maybe_rescale_pixel_size(
            potential.voxel_size * fourier_projected_potential_without_postprocess,
            potential.voxel_size,
            instrument_config.pixel_size,
            is_real=False,
            shape_in_real_space=instrument_config.padded_shape,
        )
