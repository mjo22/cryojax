"""
Methods for integrating the scattering potential directly onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import jax.numpy as jnp
from equinox import AbstractVar, error_if, Module
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

    pixel_rescaling_method: AbstractVar[Optional[str]]

    def _convert_raw_image_to_integrated_potential(
        self,
        fourier_integrated_potential_without_postprocess,
        potential,
        instrument_config,
    ):
        """Return the integrated potential in fourier space at the
        `instrument_config.pixel_size` and the `instrument_config.padded_shape.`
        """
        if self.pixel_rescaling_method is None:
            fourier_integrated_potential = error_if(
                potential.voxel_size * fourier_integrated_potential_without_postprocess,
                ~jnp.isclose(potential.voxel_size, instrument_config.pixel_size),
                f"Tried to use {type(self).__name__} with `{type(potential).__name__}."
                "voxel_size != InstrumentConfig.pixel_size`. If this is true, then "
                f"`{type(self).__name__}.pixel_rescaling_method` must not be set to "
                f"`None`. Try setting `{type(self).__name__}.pixel_rescaling_method = "
                "'bicubic'`.",
            )
            return fourier_integrated_potential
        else:
            fourier_integrated_potential = maybe_rescale_pixel_size(
                potential.voxel_size * fourier_integrated_potential_without_postprocess,
                potential.voxel_size,
                instrument_config.pixel_size,
                is_real=False,
                shape_in_real_space=instrument_config.padded_shape,
            )
            return fourier_integrated_potential
