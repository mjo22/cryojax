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

import mrcfile, os
import numpy as np
import jax.numpy as jnp
from typing import Any
from ..utils import fftfreqs, fft
from ..core import Array, ArrayLike


def load_grid_as_cloud(filename: str, **kwargs: Any) -> dict:
    """
    Read a 3D template on a cartesian grid
    to a ``Cloud``.

    Parameters
    ----------
    filename : `str`
        Path to template.
    kwargs :
        Keyword arguments passed to
        ``cryojax.io.coordinatize_voxels``.

    Returns
    -------
    cloud : `dict`
        Electron density in a point cloud representation,
        generated from a 3D voxel template. By default,
        voxels with zero density are masked.
        Instantiates a ``cryojax.simulator.ElectronCloud``
    """
    # Load template
    filename = os.path.abspath(filename)
    template, voxel_size = load_mrc(filename)
    # Load density and coordinates
    density, coordinates = coordinatize_voxels(template, voxel_size, **kwargs)
    # Gather fields to instantiate an ElectronCloud
    cloud = dict(
        density=density,
        coordinates=coordinates,
        voxel_size=voxel_size,
        filename=filename,
        config=kwargs,
    )

    return cloud


def load_fourier_grid(filename: str, **kwargs: Any) -> dict:
    """
    Read a 3D template on a cartesian grid
    to a ``Cloud``.

    Parameters
    ----------
    filename : `str`
        Path to template.

    Returns
    -------
    voxels : `dict`
        3D electron density in a 3D voxel grid representation.
        Instantiates a ``cryojax.simulator.ElectronGrid``
    """
    # Load template
    filename = os.path.abspath(filename)
    template, voxel_size = load_mrc(filename)
    # Load density and coordinates
    density = fft(template)
    coordinates = jnp.array(fftfreqs(template.shape, voxel_size, real=False))
    # Gather fields to instantiate an ElectronGrid
    voxels = dict(
        density=density,
        coordinates=coordinates,
        voxel_size=voxel_size,
        filename=filename,
        config=kwargs,
    )

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
    voxel_size :
        The voxel_size in each dimension, stored
        in the MRC file.
    """
    with mrcfile.open(filename) as mrc:
        template = np.asarray(mrc.data, dtype=float)
        voxel_size = np.asarray(
            [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=float
        )

    return template, voxel_size


def coordinatize_voxels(
    template: ArrayLike,
    voxel_size: ArrayLike,
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
    voxel_size : shape `(3,)`
        Voxel size of the template.
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
    R = fftfreqs(shape, voxel_size, real=True)
    for i in range(ndim):
        if mask:
            coords[..., i] = R[..., i].ravel()[nonzero]
        else:
            coords[..., i] = R[..., i].ravel()

    return jnp.array(density), jnp.array(coords)
