from abc import abstractmethod

from equinox import Module
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
        voltage_in_kilovolts: Float[Array, ""] | float = 300.0,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        raise NotImplementedError


class AbstractTransferTheory(Module, strict=True):
    """Base class for a transfer theory."""

    @abstractmethod
    def __call__(
        self,
        array: (
            Complex[
                Array,
                "{instrument_config.padded_y_dim} "
                "{instrument_config.padded_x_dim//2+1}",
            ]
            | Complex[
                Array,
                "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
            ]
        ),
        instrument_config: InstrumentConfig,
    ) -> (
        Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
        | Complex[
            Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
        ]
    ):
        """Pass a quantity through the transfer theory."""
        raise NotImplementedError
