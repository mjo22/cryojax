"""
Coordinate functionality in cryojax.
"""

from __future__ import annotations

__all__ = [
    "CoordinateList",
    "CoordinateGrid",
    "FrequencyGrid",
    "FrequencySlice",
]

from abc import abstractmethod
from jaxtyping import ArrayLike, Array, Float
from typing import TypeVar, Optional
from typing_extensions import overload

import equinox as eqx
import jax.numpy as jnp

from ..typing import (
    CloudCoords3D,
    CloudCoords2D,
    ImageCoords,
    VolumeCoords,
    VolumeSliceCoords,
)
from ..image import make_coordinates, make_frequencies


CoordinateType = TypeVar("CoordinateType", bound="Coordinates")
"""Type hint for a coordinate-like object."""


class Coordinates(eqx.Module):
    """
    A base class that wraps a coordinate array.
    """

    _coordinates: Array

    @abstractmethod
    def __init__(self, coordinates: Array):
        """Set the coordinate array"""
        self._coordinates = coordinates

    def get(self):
        """Get the coordinates."""
        return self._coordinates

    def __mul__(self: CoordinateType, arr: ArrayLike) -> CoordinateType:
        cls = type(self)
        return cls(self._coordinates * jnp.asarray(arr))

    def __rmul__(self: CoordinateType, arr: ArrayLike) -> CoordinateType:
        cls = type(self)
        return cls(jnp.asarray(arr) * self._coordinates)

    def __truediv__(self: CoordinateType, arr: ArrayLike) -> CoordinateType:
        cls = type(self)
        return cls(self._coordinates / jnp.asarray(arr))

    def __rtruediv__(self: CoordinateType, arr: ArrayLike) -> CoordinateType:
        cls = type(self)
        return cls(jnp.asarray(arr) / self._coordinates)


class CoordinateList(Coordinates):
    """
    A Pytree that wraps a coordinate list.
    """

    _coordinates: CloudCoords3D | CloudCoords2D = eqx.field(
        converter=jnp.asarray
    )

    def __init__(self, coordinate_list: CloudCoords2D | CloudCoords3D):
        self._coordinates = coordinate_list


class CoordinateGrid(Coordinates):
    """
    A Pytree that wraps a coordinate grid.
    """

    _coordinates: ImageCoords | VolumeCoords = eqx.field(converter=jnp.asarray)

    @overload
    def __init__(
        self,
        coordinate_grid: ImageCoords | VolumeCoords,
        *,
        shape: Optional[tuple[int, int] | tuple[int, int, int]],
        grid_spacing: float,
    ):
        ...

    @overload
    def __init__(
        self,
        coordinate_grid: None,
        *,
        shape: tuple[int, int] | tuple[int, int, int],
        grid_spacing: float = 1.0,
    ):
        ...

    def __init__(
        self,
        coordinate_grid: Optional[ImageCoords | VolumeCoords] = None,
        *,
        shape: Optional[tuple[int, int] | tuple[int, int, int]] = None,
        grid_spacing: float = 1.0,
    ):
        if coordinate_grid is not None:
            self._coordinates = coordinate_grid
        elif shape is not None:
            self._coordinates = make_coordinates(shape, grid_spacing)
        else:
            raise ValueError("Must either pass a coordinate grid or a shape.")


class FrequencyGrid(Coordinates):
    """
    A Pytree that wraps a frequency grid.
    """

    _coordinates: ImageCoords | VolumeCoords = eqx.field(converter=jnp.asarray)

    @overload
    def __init__(
        self,
        frequency_grid: ImageCoords | VolumeCoords,
        *,
        shape: Optional[tuple[int, int] | tuple[int, int, int]],
        grid_spacing: float,
        half_space: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        frequency_grid: None,
        *,
        shape: tuple[int, int] | tuple[int, int, int],
        grid_spacing: float = 1.0,
        half_space: bool = True,
    ):
        ...

    def __init__(
        self,
        frequency_grid: Optional[ImageCoords | VolumeCoords] = None,
        *,
        shape: Optional[tuple[int, int] | tuple[int, int, int]] = None,
        grid_spacing: float = 1.0,
        half_space: bool = True,
    ):
        if frequency_grid is not None:
            self._coordinates = frequency_grid
        elif shape is not None:
            self._coordinates = make_frequencies(
                shape, grid_spacing, half_space=half_space
            )
        else:
            raise ValueError("Must either pass a coordinate grid or a shape.")


class FrequencySlice(Coordinates):
    """
    A Pytree that wraps a frequency grid.
    """

    _coordinates: VolumeSliceCoords = eqx.field(converter=jnp.asarray)

    @overload
    def __init__(
        self,
        frequency_slice: VolumeSliceCoords,
        *,
        shape: Optional[tuple[int, int]],
        grid_spacing: float,
        half_space: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        frequency_slice: None,
        *,
        shape: tuple[int, int],
        grid_spacing: float = 1.0,
        half_space: bool = True,
    ):
        ...

    def __init__(
        self,
        frequency_slice: Optional[VolumeSliceCoords] = None,
        *,
        shape: Optional[tuple[int, int]] = None,
        grid_spacing: float = 1.0,
        half_space: bool = True,
    ):
        if frequency_slice is not None:
            self._coordinates = frequency_slice
        elif shape is not None:
            frequency_slice = make_frequencies(
                shape, grid_spacing, half_space=half_space
            )
            frequency_slice = jnp.expand_dims(
                jnp.pad(
                    frequency_slice,
                    ((0, 0), (0, 0), (0, 1)),
                    mode="constant",
                    constant_values=0.0,
                ),
                axis=2,
            )
            self._coordinates = frequency_slice
        else:
            raise ValueError("Must either pass a coordinate grid or a shape.")
