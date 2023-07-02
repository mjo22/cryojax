"""
Routines to model image formation.
"""

__all__ = ["project"]

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax_2dtm.types import Array
from jax_2dtm.utils import nufft


# @partial(jit, static_argnames="shape")
def project(
    density: Array,
    coords: Array,
    box_shape: tuple[int, int, int],
    pixel_size: float,
    eps: float = 1e-6,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    Arguments
    ---------
    density : shape `(N,)`
        Point cloud volume density.
    coords : shape `(N, 3)`
        Coordinates of ``density``.
    box_shape :
        Shape of the 3D voxel domain that the
        density occupies. ``(shape[0], shape[1])``
        can be arbitrary (for example, they can be
        the shape of the micrograph), while ``shape[2]``
        should be the boxsize of the 3D reference volume.
    pixel_size :
        Size of each voxel, in dimensions of length.

    Returns
    -------
    image :
        The output image in the fourier domain.
    """
    box_size = jnp.array(box_shape, dtype=float) * pixel_size
    image_shape = (box_shape[0], box_shape[1], int(1))
    image = nufft(density, coords, box_size, image_shape, eps=eps)[:, :, 0]

    return image


# @partial(jit, static_argnames=["shape"])
def project_as_histogram(
    density: Array, coords: Array, shape: tuple[int, int, int]
) -> Array:
    """
    Project 3D volume onto imaging plane
    by computing a histogram.

    Arguments
    ----------
    density : shape `(N, 3)`
        3D volume.
    coords : shape `(N, 3)`
        Coordinate system.
    shape :
        A tuple denoting the shape of the output image, given
        by ``(N2, N3)``
    Returns
    -------
    projection : shape `(N2, N3)`
        Projection of volume onto imaging plane,
        which is taken to be axis 0.
    """
    N2, N3 = shape[0], shape[1]
    # Round coordinates for binning
    rounded_coords = jnp.rint(coords).astype(int)
    # Shift coordinates back to zero in the corner, rather than center
    y_coords, z_coords = (
        rounded_coords[:, 1] + N2 // 2,
        rounded_coords[:, 2] + N3 // 2,
    )
    # Bin values on the same y-z plane
    flat_coords = jnp.ravel_multi_index(
        (y_coords, z_coords), (N2, N3), mode="clip"
    )
    projection = jnp.bincount(
        flat_coords, weights=density, length=N2 * N3
    ).reshape((N2, N3))

    return projection
