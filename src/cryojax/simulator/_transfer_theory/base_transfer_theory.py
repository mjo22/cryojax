from abc import abstractmethod

from equinox import Module
from jaxtyping import Array, Complex, Float


class AbstractTransferFunction(Module, strict=True):
    """An abstract base class for a transfer function."""

    @abstractmethod
    def compute_phase_shifts_from_instrument(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        voltage_in_kilovolts: Float[Array, ""] | float = 300.0,
    ) -> Float[Array, "y_dim x_dim"]:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        voltage_in_kilovolts: Float[Array, ""] | float = 300.0,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        raise NotImplementedError
