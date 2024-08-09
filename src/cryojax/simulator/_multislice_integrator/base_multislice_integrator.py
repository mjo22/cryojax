from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import jax.numpy as jnp
from equinox import AbstractVar, error_if, Module
from jaxtyping import Array, Complex

from ...image import maybe_rescale_pixel_size
from .._instrument_config import InstrumentConfig


PotentialT = TypeVar("PotentialT")


class AbstractMultisliceIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a multi-slice integration scheme."""

    pixel_rescaling_method: AbstractVar[Optional[str]]

    @abstractmethod
    def compute_wavefunction_at_exit_plane(
        self,
        potential: PotentialT,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        raise NotImplementedError

    def _postprocess_exit_wave(
        self,
        exit_wave: Complex[Array, "_ _"],
        potential,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Rescale the pixel size of the exit wave, if necessary
        if self.pixel_rescaling_method is None:
            exit_wave = error_if(
                exit_wave,
                ~jnp.isclose(potential.voxel_size, instrument_config.pixel_size),
                f"Tried to use {type(self).__name__} with `{type(potential).__name__}."
                "voxel_size != InstrumentConfig.pixel_size`. If this is true, then "
                f"`{type(self).__name__}.pixel_rescaling_method` must not be set to "
                f"`None`. Try setting `{type(self).__name__}.pixel_rescaling_method = "
                "'bicubic'`.",
            )
        else:
            exit_wave = maybe_rescale_pixel_size(
                exit_wave,
                potential.voxel_size,
                instrument_config.pixel_size,
                is_real=True,
            )
        # Resize the image to match the InstrumentConfig.padded_shape
        if instrument_config.padded_shape != exit_wave.shape:
            exit_wave = instrument_config.crop_or_pad_to_padded_shape(exit_wave)

        return exit_wave
