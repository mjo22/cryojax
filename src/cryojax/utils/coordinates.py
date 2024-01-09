"""
Routines to compute and operate on coordinate systems.
"""

__all__ = [
    "make_coordinates",
    "make_frequencies",
    "cartesian_to_polar",
    "flatten_and_coordinatize",
]

from typing import Union, Any, Optional
from jaxtyping import Array, Float

import jax.numpy as jnp

from ..typing import RealImage, RealVolume, Image, ImageCoords


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
    half_space: bool = False,
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


def flatten_and_coordinatize(
    template: Union[RealVolume, RealImage],
    voxel_size: float,
    mask_zeros: bool = True,
    indexing="xy",
    **kwargs: Any,
) -> tuple[Float[Array, "N"], Float[Array, "N ndim"]]:
    """
    Returns 3D volume or 2D image and its coordinate system.
    By default, coordinates are shape ``(N, ndim)``, where
    ``ndim = template.ndim`` and ``N = N1*N2*N3 - M`` or
    ``N = N2*N3 - M``, where ``M`` is a number of points
    close to zero that are masked out. The coordinate system
    is set with dimensions of length with zero in the center.

    Parameters
    ----------
    template : `Array`, shape `(N1, N2, N3)` or `(N1, N2)`
        3D volume or 2D image on a cartesian grid.
    voxel_size : float
        Pixel size or voxel size of the template.
    mask_zeros : `bool`
        If ``True``, run template through ``jax.numpy.isclose``
        to remove coordinates with zero electron density.
    kwargs
        Keyword arguments passed to ``numpy.isclose``.
        Disabled for ``mask_zeros = False``.

    Returns
    -------
    flat_and_masked : `Array`, shape `(N, ndim)`
        Flattened and masked volume or image.
    coordinate_list : `Array`, shape `(N, ndim)`
        Cartesian coordinate system.
    """
    template = jnp.asarray(template)
    ndim, shape = template.ndim, template.shape
    # Mask out points where the electron density is close
    # to zero.
    flat = template.ravel()
    if mask_zeros:
        nonzero = jnp.where(~jnp.isclose(flat, 0.0, **kwargs))
        flat_and_masked = flat[nonzero]
    else:
        nonzero = True
        flat_and_masked = flat

    # Create coordinate buffer
    N = flat_and_masked.size
    coordinate_list = jnp.zeros((N, ndim))

    # Generate rectangular grid and fill coordinate array.
    R = make_coordinates(shape, voxel_size, indexing=indexing)
    for i in range(ndim):
        if mask_zeros:
            coordinate_list = coordinate_list.at[..., i].set(
                R[..., i].ravel()[nonzero]
            )
        else:
            coordinate_list = coordinate_list.at[..., i].set(R[..., i].ravel())

    return flat_and_masked, coordinate_list


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
