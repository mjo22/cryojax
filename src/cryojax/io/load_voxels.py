"""
Routines for loading voxel-based electron densities.
"""

from __future__ import annotations

__all__ = ["load_mrc", "load_voxel_cloud", "load_fourier_grid"]

import mrcfile, os
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
from typing import Any
from ..utils import fftn, pad, make_frequencies, flatten_and_coordinatize


def load_voxel_cloud(filename: str, **kwargs: Any) -> dict[str, Array]:
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
        ``cryojax.utils.coordinates.flatten_and_coordinatize``.

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
    # Change how template sits in the box to match cisTEM
    # Ideally we would have this read in from the MRC and be the
    # same for all I/O methods. However, the algorithms used all
    # have their own xyz conventions. This is made to match
    # jax-finufft.
    template = jnp.transpose(template, axes=[1, 2, 0])
    # Load flattened density and coordinates
    density, coordinates = flatten_and_coordinatize(
        template, voxel_size, **kwargs
    )
    # Gather fields to instantiate an ElectronCloud
    cloud = dict(weights=density, coordinates=coordinates)

    return cloud


def load_fourier_grid(filename: str, pad_scale: float = 1.0) -> dict[str, Any]:
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
    # Change how template sits in box to match cisTEM
    template = jnp.transpose(template, axes=[2, 1, 0])
    # Pad template
    padded_shape = tuple([int(s * pad_scale) for s in template.shape])
    template = pad(template, padded_shape)
    # Load density and coordinates
    density = fftn(template)
    coordinates = make_frequencies(template.shape, voxel_size)
    # Get central z slice
    coordinates = jnp.expand_dims(coordinates[:, :, 0, :], axis=2)
    # Gather fields to instantiate an ElectronGrid
    voxels = dict(weights=density, coordinates=coordinates)

    return voxels


def load_mrc(filename: str) -> tuple[np.ndarray, float]:
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
    # Read MRC
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
    ), "MRC file must set the voxel size."
    assert all(
        voxel_size == voxel_size[0]
    ), "Voxel size must be same in all dimensions."

    return data, voxel_size[0]
