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
from ..utils import fftfreqs, fft
from ..core import Array, ArrayLike


def load_grid_as_cloud(
    filename: str, config: ImageConfig, real: bool = True, **kwargs: Any
) -> Cloud:
    """
    Read a 3D template on a cartesian grid
    to a ``Cloud``.

    Parameters
    ----------
    filename : `str`
        Path to template.
    config : `jax_2dtm.simulator.ImageConfig`
        Image configuration.
    real : `bool`
        If ``True``, load volume in real space.
        If ``False``, load in Fourier space.
    kwargs :
        Keyword arguments passed to
        ``jax_2dtm.io.coordinatize``.

    Returns
    -------
    cloud : `jax_2dtm.simulator.Cloud`
        Point cloud generated from the 3D template.
    """
    # Load template
    template = load_mrc(filename)
    # Determine box_size, taking the z depth to be the longest
    # box dimesion
    depth = max(template.shape)
    box_size = (
        jnp.array((*config.padded_shape, depth), dtype=float)
        * config.pixel_size
    )
    # Instantiate a Cloud
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
    flatten: bool = True,
    **kwargs: Any,
) -> tuple[Array, ...]:
    """
    Returns 3D volume or 2D image and its coordinate system.
    By default, coordinates are shape ``(N, ndim)``, where
    ``ndim = template.ndim`` and ``N = N1*N2*N3 - M`` or
    ``N = N2*N3 - M``, where ``M`` is a number of points
    close to zero that are masked out. The coordinate system
    has dimensions of length with zero in the center.

    Parameters
    ----------
    template : shape `(N1, N2, N3)` or `(N1, N2)`
        3D volume or 2D image on a cartesian grid.
    pixel_size : `float`
        Camera pixel size.
    real : `bool`
        If ``True``, return density and coordinate
        system in real space. If ``False``, return density
        coordinates in Fourier space.
    flatten : `bool`
        If ``True``, return flattened density and coordinates,
        which allows for optional keyword arguments for masking out
        zero density values. If ``False``, do not flatten or
        mask the volume.
    kwargs
        Keyword arguments passed to ``np.isclose``.
        Disabled for ``flatten = False``.

    Returns
    -------
    density : shape `(N, ndim)` or `(N1, N2, ...)`
        Volume or image.
    coords : shape `(N, ndim)` or `(N1, N2, ..., ndim)`
        Cartesian coordinate system.
    """
    ndim, shape = template.ndim, template.shape

    # Load template in real or fourier space
    density = template if real else fft(template)
    # Flatten to point cloud or keep as is
    if flatten:
        # Mask out points where the electron density is close
        # to zero.
        flat = density.ravel()
        mask = np.where(~np.isclose(flat, 0.0, **kwargs))
        density = flat[mask]
    else:
        mask = True

    # Create coordinate buffer
    N = density.size
    coords = np.zeros((N, ndim)) if flatten else np.zeros((*shape, ndim))

    # Generate rectangular grid and fill coordinate array
    R = fftfreqs(shape, pixel_size, real=real)
    for i in range(ndim):
        if flatten:
            coords[..., i] = R[..., i].ravel()[mask]
        else:
            coords[..., i] = R[..., i]

    return jnp.array(density), jnp.array(coords)
