"""
Functions for creating and operating on coordinate systems.
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Inexact


def make_coordinates(
    shape: tuple[int, ...], grid_spacing: float | Float[np.ndarray, ""] = 1.0
) -> Float[Array, "*shape ndim"]:
    """
    Create a real-space cartesian coordinate system on a grid.

    Arguments
    ---------
    shape :
        Shape of the voxel grid, with
        ``ndim = len(shape)``.
    grid_spacing :
        The grid spacing, in units of length.

    Returns
    -------
    coordinate_grid :
        Cartesian coordinate system in real space.
    """
    coordinate_grid = _make_coordinates_or_frequencies(
        shape, grid_spacing=grid_spacing, real_space=True
    )
    return coordinate_grid


def make_frequencies(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] = 1.0,
    half_space: bool = True,
) -> Float[Array, "*shape ndim"]:
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
    )
    return frequency_grid


def cartesian_to_polar(
    freqs: Float[Array, "y_dim x_dim 2"], square: bool = False
) -> tuple[Inexact[Array, "y_dim x_dim"], Inexact[Array, "y_dim x_dim"]]:
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
    grid_spacing: float | Float[np.ndarray, ""] = 1.0,
    real_space: bool = False,
    half_space: bool = True,
) -> Float[Array, "*shape ndim"]:
    ndim = len(shape)
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
                rfftfreq = False if idx < ndim - 1 else True
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, real_space, rfftfreq
            )
        coords1D.append(c1D)
    if ndim == 2:
        y, x = coords1D
        xv, yv = jnp.meshgrid(x, y, indexing="xy")
        coords = jnp.stack([xv, yv], axis=-1)
    elif ndim == 3:
        z, y, x = coords1D
        xv, yv, zv = jnp.meshgrid(x, y, z, indexing="xy")
        xv, yv, zv = [
            jnp.transpose(rv, axes=[2, 0, 1]) for rv in [xv, yv, zv]
        ]  # Change axis ordering to [z, y, x]
        coords = jnp.stack([xv, yv, zv], axis=-1)
    else:
        raise ValueError(
            "Only 2D and 3D coordinate grids are supported. "
            f"Tried to create a grid of shape {shape}."
        )

    return coords


def _make_coordinates_or_frequencies_1d(
    size: int,
    grid_spacing: float | Float[np.ndarray, ""],
    real_space: bool = False,
    rfftfreq: Optional[bool] = None,
) -> Float[Array, " size"]:
    """One-dimensional coordinates in real or fourier space"""
    if real_space:
        make_1d = lambda size, dx: jnp.fft.fftshift(jnp.fft.fftfreq(size, 1 / dx)) * size
    else:
        if rfftfreq is None:
            raise ValueError("Argument rfftfreq cannot be None if real_space=False.")
        else:
            fn = jnp.fft.rfftfreq if rfftfreq else jnp.fft.fftfreq
            make_1d = lambda size, dx: fn(size, grid_spacing)

    return make_1d(size, grid_spacing)
