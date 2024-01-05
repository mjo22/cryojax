"""
Routines to compute and operate on coordinate systems.
"""

__all__ = [
    "make_coordinates",
    "make_frequencies",
    "cartesian_to_polar",
    "flatten_and_coordinatize",
]

from typing import Union, Any
from jaxtyping import Array, Float

import jax.numpy as jnp

from ..typing import RealImage, RealVolume, Image, ImageCoords


def make_coordinates(*args: Any, **kwargs: Any) -> Float[Array, "... D"]:
    """
    Wraps ``_make_coordinates_or_frequencies`` for ``real = True``.
    """
    return _make_coordinates_or_frequencies(*args, real=True, **kwargs)


def make_frequencies(*args: Any, **kwargs: Any) -> Float[Array, "... D"]:
    """
    Wraps ``_make_coordinates_or_frequencies`` for ``real = False``.
    """
    return _make_coordinates_or_frequencies(*args, real=False, **kwargs)


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
        Disabled for ``mask = False``.

    Returns
    -------
    density : `Array`, shape `(N, ndim)`
        Volume or image.
    coords : `Array`, shape `(N, ndim)`
        Cartesian coordinate system.
    """
    template = jnp.asarray(template)
    ndim, shape = template.ndim, template.shape
    # Mask out points where the electron density is close
    # to zero.
    flat = template.ravel()
    if mask_zeros:
        nonzero = jnp.where(~jnp.isclose(flat, 0.0, **kwargs))
        density = flat[nonzero]
    else:
        nonzero = True
        density = flat

    # Create coordinate buffer
    N = density.size
    coords = jnp.zeros((N, ndim))

    # Generate rectangular grid and fill coordinate array.
    R = make_coordinates(shape, voxel_size, indexing=indexing)
    for i in range(ndim):
        if mask_zeros:
            coords = coords.at[..., i].set(R[..., i].ravel()[nonzero])
        else:
            coords = coords.at[..., i].set(R[..., i].ravel())

    return density, coords


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
    real: bool = False,
    indexing: str = "xy",
) -> Float[Array, "... D"]:
    """
    Create a cartesian coordinate system on a grid.
    This can be used for real and fourier space.
    If used for fourier space, the
    zero-frequency component is in the beginning.

    Arguments
    ---------
    shape :
        Shape of the voxel grid, with
        ``ndim = len(shape)``.
    pixel_size :
        Image pixel size.
    real :
        Choose whether to create coordinate system
        in real or fourier space.

    Returns
    -------
    coords :
        2D or 3D cartesian coordinate system.
    """
    ndim = len(shape)
    shape = (*shape[:2][::-1], *shape[2:]) if indexing == "xy" else shape
    coords1D = [
        _make_coordinates_or_frequencies_1d(shape[idx], grid_spacing, real) for idx in range(ndim)
    ]
    coords = jnp.stack(jnp.meshgrid(*coords1D, indexing=indexing), axis=-1)

    return coords


def _make_coordinates_or_frequencies_1d(size: int, grid_spacing: float, real: bool = False) -> Array:
    """One-dimensional coordinates in real or fourier space"""
    coordinate_fn = (
        lambda size, dx: jnp.fft.fftshift(jnp.fft.fftfreq(size, 1 / dx)) * size
        if real
        else jnp.fft.fftfreq(size, grid_spacing)
    )
    return coordinate_fn(size, grid_spacing)
