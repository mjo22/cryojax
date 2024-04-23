from abc import abstractmethod
from typing import Optional

from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float

from ...image.operators import (
    AbstractFourierOperator,
)
from .._instrument_config import InstrumentConfig


class AbstractTransferFunction(AbstractFourierOperator, strict=True):
    """An abstract base class for a transfer function."""

    @abstractmethod
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        wavelength_in_angstroms: Optional[Float[Array, ""] | float] = None,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        raise NotImplementedError


class AbstractTransferTheory(Module, strict=True):
    """Base class for a transfer theory."""

    transfer_function: AbstractVar[AbstractTransferFunction]

    @abstractmethod
    def __call__(
        self,
        fourier_phase_or_wavefunction_in_exit_plane: (
            Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
            | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
        ),
        config: InstrumentConfig,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> (
        Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
        | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        """Pass an image through the transfer theory."""
        raise NotImplementedError
