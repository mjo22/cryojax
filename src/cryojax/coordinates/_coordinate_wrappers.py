"""
Coordinate abstractions.
"""

from abc import abstractmethod
from typing import Any
from typing_extensions import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar
from jaxtyping import Array, Float

from ._coordinate_functions import make_coordinates, make_frequencies


class AbstractCoordinates(eqx.Module, strict=True):
    """
    A base class that wraps a coordinate array.
    """

    array: AbstractVar[Any]

    @abstractmethod
    def get(self) -> Any:
        """Get the coordinates."""
        raise NotImplementedError

    def __mul__(
        self, real_number: float | Float[np.ndarray, ""] | Float[Array, ""]
    ) -> Self:
        # The following line seems to be required for differentiability with
        # respect to arr
        rescaled_array = jnp.where(
            self.array != 0.0, self.array * jnp.asarray(real_number), 0.0
        )
        return eqx.tree_at(lambda x: x.array, self, rescaled_array)

    def __rmul__(
        self, real_number: float | Float[np.ndarray, ""] | Float[Array, ""]
    ) -> Self:
        rescaled_array = jnp.where(
            self.array != 0.0, jnp.asarray(real_number) * self.array, 0.0
        )
        return eqx.tree_at(lambda x: x.array, self, rescaled_array)

    def __truediv__(
        self, real_number: float | Float[np.ndarray, ""] | Float[Array, ""]
    ) -> Self:
        rescaled_array = jnp.where(
            self.array != 0.0, self.array / jnp.asarray(real_number), 0.0
        )
        return eqx.tree_at(lambda x: x.array, self, rescaled_array)


class CoordinateList(AbstractCoordinates, strict=True):
    """
    A Pytree that wraps a coordinate list.
    """

    array: Float[Array, "size 3"] | Float[Array, "size 2"] = eqx.field(
        converter=jnp.asarray
    )

    def __init__(self, coordinate_list: Float[Array, "size 2"] | Float[Array, "size 3"]):
        self.array = coordinate_list

    def get(self) -> Float[Array, "size 3"] | Float[Array, "size 2"]:
        return self.array


class CoordinateGrid(AbstractCoordinates, strict=True):
    """
    A Pytree that wraps a coordinate grid.
    """

    array: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"] = (
        eqx.field(converter=jnp.asarray)
    )

    def __init__(
        self,
        shape: tuple[int, ...],
        grid_spacing: float | Float[np.ndarray, ""] = 1.0,
    ):
        self.array = make_coordinates(shape, grid_spacing)

    def get(
        self,
    ) -> Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]:
        return self.array


class FrequencyGrid(AbstractCoordinates, strict=True):
    """
    A Pytree that wraps a frequency grid.
    """

    array: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"] = (
        eqx.field(converter=jnp.asarray)
    )

    def __init__(
        self,
        shape: tuple[int, ...],
        grid_spacing: float | Float[np.ndarray, ""] = 1.0,
        half_space: bool = True,
    ):
        self.array = make_frequencies(shape, grid_spacing, half_space=half_space)

    def get(
        self,
    ) -> Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]:
        return self.array


class FrequencySlice(AbstractCoordinates, strict=True):
    """
    A Pytree that wraps a frequency slice.

    Unlike a `FrequencyGrid`, a `FrequencySlice` has the zero frequency
    component in the center.
    """

    array: Float[Array, "1 y_dim x_dim 3"] = eqx.field(converter=jnp.asarray)

    def __init__(
        self,
        shape: tuple[int, int],
        grid_spacing: float | Float[np.ndarray, ""] = 1.0,
        half_space: bool = True,
    ):
        frequency_slice = make_frequencies(shape, grid_spacing, half_space=half_space)
        if half_space:
            frequency_slice = jnp.fft.fftshift(frequency_slice, axes=(0,))
        else:
            frequency_slice = jnp.fft.fftshift(frequency_slice, axes=(0, 1))
        frequency_slice = jnp.expand_dims(
            jnp.pad(
                frequency_slice,
                ((0, 0), (0, 0), (0, 1)),
                mode="constant",
                constant_values=0.0,
            ),
            axis=0,
        )
        self.array = frequency_slice

    def get(self) -> Float[Array, "1 y_dim x_dim 3"]:
        return self.array
