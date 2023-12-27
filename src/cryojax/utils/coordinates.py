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
import numpy as np

from ..typing import RealImage, RealVolume, Image, ImageCoords


def make_coordinates(*args: Any, **kwargs: Any) -> Float[Array, "... D"]:
    """
    Wraps ``_fftfreqs`` for ``real = True``.
    """
    return _fftfreqs(*args, real=True, **kwargs)


def make_frequencies(*args: Any, **kwargs: Any) -> Float[Array, "... D"]:
    """
    Wraps ``_fftfreqs`` for ``real = False``.
    """
    return _fftfreqs(*args, real=False, **kwargs)


def flatten_and_coordinatize(
    template: Union[RealVolume, RealImage],
    resolution: float,
    mask: bool = True,
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
    template : `np.ndarray`, shape `(N1, N2, N3)` or `(N1, N2)`
        3D volume or 2D image on a cartesian grid.
    resolution : float
        Resolution of the template.
    mask : `bool`
        If ``True``, run template through ``numpy.isclose``
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
    if mask:
        nonzero = np.where(~np.isclose(flat, 0.0, **kwargs))
        density = flat[nonzero]
    else:
        nonzero = True
        density = flat

    # Create coordinate buffer
    N = density.size
    coords = np.zeros((N, ndim))

    # Generate rectangular grid and fill coordinate array.
    R = _fftfreqs(shape, resolution, real=True, indexing=indexing)
    for i in range(ndim):
        if mask:
            coords[..., i] = R[..., i].ravel()[nonzero]
        else:
            coords[..., i] = R[..., i].ravel()

    return jnp.array(density), jnp.array(coords)


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


def _fftfreqs(
    shape: tuple[int, ...],
    pixel_size: Union[float, np.ndarray] = 1.0,
    real: bool = False,
    indexing: str = "xy",
) -> Float[Array, "... D"]:
    """
    Create a cartesian coordinate system on a grid.
    This can be used for real and fourier space.
    If used for fourier space, the
    zero-frequency component is in the beginning.

    Note that the current implementations create coordinates in
    ``numpy``, then transfer to ``jax``. Therefore, these should
    not be run inside jit compilation.

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
    if isinstance(pixel_size, (np.floating, float)):
        pixel_size = ndim * [pixel_size]
    else:
        pixel_size = list(pixel_size)
    assert len(pixel_size) == ndim
    coords1D = [
        _fftfreqs1d(shape[idx], pixel_size[idx], real) for idx in range(ndim)
    ]
    coords = np.stack(np.meshgrid(*coords1D, indexing=indexing), axis=-1)

    return jnp.asarray(coords)


def _fftfreqs1d(s: int, pixel_size: float, real: bool = False) -> np.ndarray:
    """One-dimensional coordinates in real or fourier space"""
    fftfreq = (
        lambda s: np.fft.fftshift(np.fft.fftfreq(s, 1 / pixel_size)) * s
        if real
        else np.fft.fftfreq(s, pixel_size)
    )
    return fftfreq(s)
