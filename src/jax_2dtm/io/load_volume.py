"""
Routines for reading 3D models into arrays.
"""

from __future__ import annotations

__all__ = ["load_mrc", "load_grid_as_cloud", "coordinatize"]

import mrcfile
import numpy as np
import jax.numpy as jnp
from typing import Any
from ..simulator import Cloud, ImageConfig
from ..utils import fftfreqs, fft, pad
from ..core import Array, ArrayLike


def load_grid_as_cloud(
    filename: str, config: ImageConfig, real: bool = True, **kwargs: Any
) -> Cloud:
    """
    Read a 3D template on a cartesian grid
    to a ``Cloud``.

    Parameters
    ----------
    filename :
        Path to template.
    config :
        Image configuration.
    real : `bool`
        If ``True``, load volume in real space.
        If ``False``, load in Fourier space.
    kwargs :
        Keyword arguments passed to
        ``jax_2dtm.simulator.coordinatize``.

    Returns
    -------
    cloud :
        Point cloud generated from the 3D template.
    """
    template = load_mrc(filename)
    depth = max(template.shape)
    box_size = (
        jnp.array((*config.padded_shape, depth), dtype=float)
        * config.pixel_size
    )
    cloud = Cloud(
        *coordinatize(template, config.pixel_size, real=real, **kwargs),
        box_size,
        real=real,
    )

    return cloud


def load_mrc(filename: str) -> ArrayLike:
    """
    Read template to ``numpy`` array.

    Parameters
    ----------
    filename :
        Path to template.

    Returns
    -------
    template :
        Model in cartesian coordinates.
    """
    with mrcfile.open(filename) as mrc:
        template = np.array(mrc.data)

    return template


def coordinatize(
    template: ArrayLike,
    pixel_size: float,
    real: bool = True,
    **kwargs: Any,
) -> tuple[Array, ...]:
    """
    Returns flattened coordinate system and 3D volume or 2D image
    of shape ``(N, ndim)``, where ``ndim = volume.ndim`` and
    ``N = N1*N2*N3 - M`` or ``N = N2*N3 - M``, where ``M`` is a
    number of points close to zero that are masked out.
    The coordinate system is in pixel coordinates with zero
    in the center.

    Parameters
    ----------
    density : shape `(N1, N2, N3)` or `(N1, N2)`
        3D volume or 2D image on a cartesian grid.
    pixel_size : `float`
        Camera pixel size.
    real : `bool`
        If ``True``, return flattened density and coordinate
        system in real space. If ``False``, return structured
        coordinates in Fourier space.
    kwargs
        Keyword arguments passed to ``np.isclose``.
        Disabled for ``real = False``.

    Returns
    -------
    density : shape `(N, ndim)` or `(N1, N2, ...)`
        Volume or image.
    coords : shape `(N, ndim)` or `(N1, N2, ..., ndim)`
        Cartesian coordinate system.
    """
    ndim, shape = template.ndim, template.shape

    # Mask out points where the electron density is close
    # to zero.
    if real:
        flat = template.ravel()
        mask = np.where(~np.isclose(flat, 0.0, **kwargs))
        density = flat[mask]
    else:
        if shape != tuple(ndim * [shape[0]]):
            template = pad(template, tuple(ndim * [max(shape)]))
            shape = template.shape
        density = fft(template)
        mask = True

    # Create coordinate buffer
    N = density.size
    coords = np.zeros((N, ndim)) if real else np.zeros((*shape, ndim))

    # Generate rectangular grid and fill coordinate array
    R = fftfreqs(shape, pixel_size, real=real)
    for i in range(ndim):
        if real:
            coords[..., i] = R[..., i].ravel()[mask]
        else:
            coords[..., i] = R[..., i]

    return jnp.array(density), jnp.array(coords)
