"""
Coordinate functionality in cryojax.
"""

from __future__ import annotations

__all__ = [
    "is_not_coordinate_array",
    "get_not_coordinate_filter_spec",
    "AbstractCoordinates",
    "CoordinateT",
    "CoordinateList",
    "CoordinateGrid",
    "FrequencyGrid",
    "FrequencySlice",
    "make_coordinates",
    "make_frequencies",
    "cartesian_to_polar",
]

from abc import abstractmethod
from jaxtyping import ArrayLike, Array, PyTree, Float
from typing import TypeVar, Optional, Any
from typing_extensions import overload

import equinox as eqx
import jax.tree_util as jtu
import jax.numpy as jnp

from ..typing import (
    Image,
    CloudCoords3D,
    CloudCoords2D,
    ImageCoords,
    VolumeCoords,
    VolumeSliceCoords,
)


CoordinateT = TypeVar("CoordinateT", bound="AbstractCoordinates")
"""Type hint for a coordinate-like object."""


#
# Filter functions
#
def is_not_coordinate_array(element: Any) -> bool:
    """Returns ``False`` if ``element`` is ``Coordinates``."""
    if isinstance(element, AbstractCoordinates):
        return False
    else:
        return eqx.is_array(element)


#
# Common filter specs
#
def get_not_coordinate_filter_spec(pytree: PyTree) -> PyTree[bool]:
    """Filter spec that is ``True`` for all arrays that are not ``Coordinates``."""
    return jtu.tree_map(
        is_not_coordinate_array,
        pytree,
        is_leaf=lambda x: isinstance(x, AbstractCoordinates),
    )


class AbstractCoordinates(eqx.Module):
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

    def __mul__(self: CoordinateT, arr: ArrayLike) -> CoordinateT:
        cls = type(self)
        return cls(self._coordinates * jnp.asarray(arr))

    def __rmul__(self: CoordinateT, arr: ArrayLike) -> CoordinateT:
        cls = type(self)
        return cls(jnp.asarray(arr) * self._coordinates)

    def __truediv__(self: CoordinateT, arr: ArrayLike) -> CoordinateT:
        cls = type(self)
        return cls(self._coordinates / jnp.asarray(arr))

    def __rtruediv__(self: CoordinateT, arr: ArrayLike) -> CoordinateT:
        cls = type(self)
        return cls(jnp.asarray(arr) / self._coordinates)


class CoordinateList(AbstractCoordinates):
    """
    A Pytree that wraps a coordinate list.
    """

    _coordinates: CloudCoords3D | CloudCoords2D = eqx.field(
        converter=jnp.asarray
    )

    def __init__(self, coordinate_list: CloudCoords2D | CloudCoords3D):
        self._coordinates = coordinate_list


class CoordinateGrid(AbstractCoordinates):
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


class FrequencyGrid(AbstractCoordinates):
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


class FrequencySlice(AbstractCoordinates):
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
        """Create a frequency slice. If not given, by default store
        with the zero frequency component in the center."""
        if frequency_slice is not None:
            self._coordinates = frequency_slice
        elif shape is not None:
            frequency_slice = make_frequencies(
                shape, grid_spacing, half_space=half_space
            )
            if half_space:
                frequency_slice = jnp.fft.fftshift(frequency_slice, axes=(0,))
            else:
                frequency_slice = jnp.fft.fftshift(
                    frequency_slice, axes=(0, 1)
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


def make_coordinates(
    shape: tuple[int, ...], grid_spacing: float = 1.0, indexing: str = "xy"
) -> Float[Array, "*shape len(shape)"]:
    """
    Create a real-space cartesian coordinate system on a grid.

    Arguments
    ---------
    shape :
        Shape of the voxel grid, with
        ``ndim = len(shape)``.
    grid_spacing :
        The grid spacing, in units of length.
    indexing :
        Either ``"xy"`` or ``"ij"``, passed to
        ``jax.numpy.meshgrid``.

    Returns
    -------
    coordinate_grid :
        Cartesian coordinate system in real space.
    """
    coordinate_grid = _make_coordinates_or_frequencies(
        shape, grid_spacing=grid_spacing, real_space=True, indexing=indexing
    )
    return coordinate_grid


def make_frequencies(
    shape: tuple[int, ...],
    grid_spacing: float = 1.0,
    half_space: bool = True,
    indexing: str = "xy",
) -> Float[Array, "*shape len(shape)"]:
    """
    Create a fourier-space cartesian coordinate system on a grid.
    The zero-frequency component is in the beginning.

    Arguments
    ---------
    shape :
        Shape of the voxel grid, with
        ``ndim = len(shape)``.
    grid_spacing :
        The grid spacing, in units of length.
    half_space :
        Return a frequency grid on the half space.
        ``shape[-1]`` is the axis on which the negative
        frequencies are omitted.
    indexing :
        Either ``"xy"`` or ``"ij"``, passed to
        ``jax.numpy.meshgrid``.

    Returns
    -------
    frequency_grid :
        Cartesian coordinate system in frequency space.
    """
    frequency_grid = _make_coordinates_or_frequencies(
        shape,
        grid_spacing=grid_spacing,
        real_space=False,
        half_space=half_space,
        indexing=indexing,
    )
    return frequency_grid


def cartesian_to_polar(
    freqs: ImageCoords, square: bool = False
) -> tuple[Image, Image]:
    """
    Convert from cartesian to polar coordinates.

    Arguments
    ---------
    freqs :
        The cartesian coordinate system.
    square :
        If ``True``, return the square of the
        radial coordinate :math:`|r|^2`. Otherwise,
        return :math:`|r|`.
    """
    theta = jnp.arctan2(freqs[..., 0], freqs[..., 1])
    k_sqr = jnp.sum(jnp.square(freqs), axis=-1)
    if square:
        return k_sqr, theta
    else:
        kr = jnp.sqrt(k_sqr)
        return kr, theta


def _make_coordinates_or_frequencies(
    shape: tuple[int, ...],
    grid_spacing: float = 1.0,
    real_space: bool = False,
    half_space: bool = True,
    indexing: str = "xy",
) -> Float[Array, "*shape len(shape)"]:
    ndim = len(shape)
    shape = (*shape[:2][::-1], *shape[2:]) if indexing == "xy" else shape
    coords1D = []
    for idx in range(ndim):
        if real_space:
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, real_space
            )
        else:
            if not half_space:
                rfftfreq = False
            else:
                if indexing == "xy" and ndim == 2:
                    rfftfreq = True if idx == 0 else False
                else:
                    rfftfreq = False if idx < ndim - 1 else True
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, real_space, rfftfreq
            )
        coords1D.append(c1D)
    coords = jnp.stack(jnp.meshgrid(*coords1D, indexing=indexing), axis=-1)

    return coords


def _make_coordinates_or_frequencies_1d(
    size: int,
    grid_spacing: float,
    real_space: bool = False,
    rfftfreq: Optional[bool] = None,
) -> Float[Array, "size"]:
    """One-dimensional coordinates in real or fourier space"""
    if real_space:
        make_1d = (
            lambda size, dx: jnp.fft.fftshift(jnp.fft.fftfreq(size, 1 / dx))
            * size
        )
    else:
        if rfftfreq is None:
            raise ValueError(
                "Argument rfftfreq cannot be None if real_space=False."
            )
        else:
            fn = jnp.fft.rfftfreq if rfftfreq else jnp.fft.fftfreq
            make_1d = lambda size, dx: fn(size, grid_spacing)

    return make_1d(size, grid_spacing)
