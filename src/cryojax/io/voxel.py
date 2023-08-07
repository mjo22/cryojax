"""
Routines for reading 3D models into arrays.
"""

from __future__ import annotations

__all__ = [
    "load_mrc",
    "load_grid_as_cloud",
    "load_fourier_grid",
    "coordinatize_voxels",
]

import mrcfile
import numpy as np
import jax.numpy as jnp
from typing import Any
from ..simulator import ElectronCloud, ElectronGrid, ImageConfig
from ..utils import fftfreqs, fft
from ..core import Array, ArrayLike


def load_grid_as_cloud(
    filename: str, config: ImageConfig, **kwargs: Any
) -> ElectronCloud:
    """
    Read a 3D template on a cartesian grid
    to a ``Cloud``.

    Parameters
    ----------
    filename : `str`
        Path to template.
    config : `cryojax.simulator.ImageConfig`
        Image configuration.
    kwargs :
        Keyword arguments passed to
        ``cryojax.io.coordinatize_voxels``.

    Returns
    -------
    cloud : `cryojax.simulator.ElectronCloud`
        Electron density in a point cloud representation,
        generated from a 3D voxel template. By default,
        voxels with zero density are masked.
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
    # Load density and coordinates
    density, coordinates = coordinatize_voxels(
        template, config.pixel_size, **kwargs
    )
    # Instantiate a Cloud
    cloud = ElectronCloud(density, coordinates, box_size)

    return cloud


def load_fourier_grid(filename: str, config: ImageConfig) -> ElectronCloud:
    """
    Read a 3D template on a cartesian grid
    to a ``Cloud``.

    Parameters
    ----------
    filename : `str`
        Path to template.
    config : `cryojax.simulator.ImageConfig`
        Image configuration.

    Returns
    -------
    voxels : `cryojax.simulator.ElectronGrid`
        3D electron density in a 3D voxel grid representation.
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
    # Load density and coordinates
    density = fft(template)
    coordinates = jnp.array(
        fftfreqs(template.shape, config.pixel_size, real=False)
    )
    # Instantiate a density in a voxel representation.
    voxels = ElectronGrid(density, coordinates, box_size)

    return voxels


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


def coordinatize_voxels(
    template: ArrayLike,
    pixel_size: float,
    mask: bool = True,
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
    mask : `bool`
        If ``True``, run template through ``numpy.isclose``
        to remove coordinates with zero electron density.
    kwargs
        Keyword arguments passed to ``numpy.isclose``.
        Disabled for ``mask = False``.

    Returns
    -------
    density : shape `(N, ndim)`
        Volume or image.
    coords : shape `(N, ndim)`
        Cartesian coordinate system.
    """
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

    # Generate rectangular grid and fill coordinate array
    R = fftfreqs(shape, pixel_size, real=True)
    for i in range(ndim):
        if mask:
            coords[..., i] = R[..., i].ravel()[nonzero]
        else:
            coords[..., i] = R[..., i].ravel()

    return jnp.array(density), jnp.array(coords)
