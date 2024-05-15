"""
Methods for integrating the scattering potential directly onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

from equinox import AbstractVar, Module
from jaxtyping import Array, Complex

from ...image import maybe_rescale_pixel_size
from .._instrument_config import InstrumentConfig
from .._potential_representation import AbstractVoxelPotential


PotentialT = TypeVar("PotentialT")
VoxelPotentialT = TypeVar("VoxelPotentialT", bound="AbstractVoxelPotential")


class AbstractPotentialIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a method of integrating a potential onto
    the exit plane.
    """

    @abstractmethod
    def compute_fourier_integrated_potential(
        self,
        potential: PotentialT,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError


class AbstractVoxelPotentialIntegrator(
    AbstractPotentialIntegrator[VoxelPotentialT], strict=True
):
    """Base class for a method of integrating a voxel-based potential."""

    pixel_rescaling_method: AbstractVar[str]

    def _convert_fourier_raw_image_to_integrated_potential(
        self,
        fourier_projected_potential_without_postprocess,
        potential,
        instrument_config,
    ):
        """Return the integrated potential in fourier space at the
        `instrument_config.pixel_size` and the `instrument_config.padded_shape.`
        """
        return maybe_rescale_pixel_size(
            potential.voxel_size * fourier_projected_potential_without_postprocess,
            potential.voxel_size,
            instrument_config.pixel_size,
            is_real=False,
            shape_in_real_space=instrument_config.padded_shape,
        )
