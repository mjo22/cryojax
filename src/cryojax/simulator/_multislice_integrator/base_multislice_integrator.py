from abc import abstractmethod
from typing import Generic, TypeVar

from equinox import Module
from jaxtyping import Array, Complex

from .._instrument_config import InstrumentConfig


PotentialT = TypeVar("PotentialT")


class AbstractMultisliceIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a multi-slice integration scheme."""

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
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Resize the image to match the InstrumentConfig.padded_shape
        if instrument_config.padded_shape != exit_wave.shape:
            exit_wave = instrument_config.crop_or_pad_to_padded_shape(exit_wave)

        return exit_wave
