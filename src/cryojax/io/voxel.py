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
from ..utils import fftfreqs, fft, pad
from ..core import Array, ArrayLike


def load_grid_as_cloud(filename: str, **kwargs: Any) -> dict:
    """
    Read a 3D template on a cartesian grid
    to a point cloud.

    This is used to instantiate ``cryojax.simulator.ElectronCloud``.

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


def load_fourier_grid(filename: str, pad_scale=1.0) -> dict:
    """
    Read a 3D template in Fourier space on a cartesian grid.

    This is used to instantiate ``cryojax.simulator.ElectronGrid``.

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
    # Pad template
    padded_shape = tuple([int(s * pad_scale) for s in template.shape])
    template = pad(template, padded_shape)
    # Load density and coordinates
    density = fft(template)
    coordinates = jnp.asarray(fftfreqs(template.shape, voxel_size, real=False))
    # Get central z slice
    _, _, N3 = density.shape
    coordinates = jnp.expand_dims(
        coordinates[:, :, N3 // 2 + N3 % 2, :], axis=2
    )
    # Gather fields to instantiate an ElectronGrid
    voxels = dict(
        density=density,
        coordinates=coordinates,
        voxel_size=voxel_size,
        filename=filename,
        config=dict(pad_scale=pad_scale),
    )

    return voxels


def load_mrc(filename: str) -> ArrayLike:
    """
    Read MRC data to ``numpy`` array.

    Parameters
    ----------
    filename : `str`
        Path to data.

    Returns
    -------
    data : `ArrayLike`, shape `(N1, N2, N3)` or `(N1, N2)`
        Model in cartesian coordinates.
    voxel_size : `ArrayLike`, shape `(3,)` or `(2,)`
        The voxel_size in each dimension, stored
        in the MRC file.
    """
    with mrcfile.open(filename) as mrc:
        data = np.asarray(mrc.data, dtype=float)
        if data.ndim == 2:
            voxel_size = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y], dtype=float
            )
        elif data.ndim == 3:
            voxel_size = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
                dtype=float,
            )
        else:
            raise NotImplementedError(
                "MRC files with 2D and 3D data are supported."
            )

    assert all(
        voxel_size != np.zeros(data.ndim)
    ), "MRC file must set voxel size"

    return data, voxel_size


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
    is set with dimensions of length with zero in the center.

    Parameters
    ----------
    template : `ArrayLike`, shape `(N1, N2, N3)` or `(N1, N2)`
        3D volume or 2D image on a cartesian grid.
    voxel_size : `ArrayLike`, shape `(3,)` or `(2,)`
        Voxel size of the template.
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
