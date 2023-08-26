"""
Routines to compute and operate on coordinate systems.
"""

__all__ = [
    "make_coordinates",
    "make_frequencies",
    "fftfreqs",
    "fftfreqs1d",
    "cartesian_to_polar",
]

from typing import Union, Any

import jax.numpy as jnp
import numpy as np

from ..core import Array, ArrayLike


def make_coordinates(*args: Any, **kwargs: Any) -> Array:
    """
    Wraps ``cryojax.utils.coordinates.fftfreqs``
    for ``real = True``.
    """
    return fftfreqs(*args, real=True, **kwargs)


def make_frequencies(*args: Any, **kwargs: Any) -> Array:
    """
    Wraps ``cryojax.utils.coordinates.fftfreqs``
    for ``real = False``.
    """
    return fftfreqs(*args, real=False, **kwargs)


def fftfreqs(
    shape: tuple[int, ...],
    pixel_size: Union[float, np.ndarray] = 1.0,
    real: bool = False,
    indexing: str = "xy",
) -> Array:
    """
    Create a radial coordinate system on a grid.
    This can be used for real and fourier space
    calculations. If used for fourier space, the
    zero-frequency component is in the center.

    Arguments
    ---------
    shape : `tuple[int, ...]`
        Shape of the voxel grid, with
        ``ndim = len(shape)``.
    pixel_size : `float` or `np.ndarray`, shape `(ndim,)`
        Image pixel size.
    real : `bool`
        Choose whether to create coordinate system
        in real or fourier space.


    Returns
    -------
    k_coords : shape `(*shape, ndim)`
        2D or 3D cartesian coordinate system with
        zero in the center.
    """
    ndim = len(shape)
    if isinstance(pixel_size, (np.floating, float)):
        pixel_size = ndim * [pixel_size]
    else:
        pixel_size = list(pixel_size)
    assert len(pixel_size) == ndim
    k_coords1D = [
        fftfreqs1d(shape[idx], pixel_size[idx], real) for idx in range(ndim)
    ]
    k_coords = np.stack(np.meshgrid(*k_coords1D, indexing=indexing), axis=-1)

    return jnp.asarray(k_coords)


def fftfreqs1d(s: int, pixel_size: float, real: bool = False) -> np.ndarray:
    """One-dimensional coordinates in real or fourier space"""
    fftfreq = (
        lambda s: np.fft.fftshift(np.fft.fftfreq(s, 1 / pixel_size)) * s
        if real
        else np.fft.fftfreq(s, pixel_size)
    )
    return fftfreq(s)


def cartesian_to_polar(
    freqs: ArrayLike, square: bool = False
) -> tuple[Array, Array]:
    """
    Convert from cartesian to polar coordinates.

    Arguments
    ---------
    freqs : `ArrayLike`, shape `(N1, N2, 2)`
        The cartesian coordinate system.
    square : `bool`, optional
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
