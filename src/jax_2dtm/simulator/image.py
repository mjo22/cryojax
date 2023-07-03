"""
Routines to model image formation.
"""

__all__ = ["project", "ImageConfig"]

import jax.numpy as jnp
from typing import NamedTuple
from .cloud import Cloud
from jax_2dtm.types import Array
from jax_2dtm.utils import nufft


class ImageConfig(NamedTuple):
    """
    Attributes
    ----------
    shape : tuple[int, int, int]
        Shape of the imaging volume in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pixel_size : float
        Size of camera pixels, in dimensions of length.
    eps : float
        Desired precision in computing the volume
        projection. See `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.
    """

    shape: tuple[int, int]
    pixel_size: float
    eps: float = 1e-6


def project(
    cloud: Cloud,
    config: ImageConfig,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    Arguments
    ---------
    cloud :
        Representation of volume point cloud.
        See ``jax_2dtm.coordinates.Cloud`` for
        more detail.
    config :
        Image configuation.

    Returns
    -------
    image :
        The output image in the fourier domain.
    """
    height, width = config.shape
    eps = config.eps
    density, coords, box_size = cloud[0:3]
    image = nufft(density, coords, box_size, (height, width, int(1)), eps=eps)[
        :, :, 0
    ]

    return image


def project_as_histogram(
    density: Array, coords: Array, shape: tuple[int, int, int]
) -> Array:
    """
    Project 3D volume onto imaging plane
    using a histogram.

    Arguments
    ----------
    density : shape `(N,)`
        3D volume.
    coords : shape `(N, 3)`
        Coordinate system.
    shape :
        A tuple denoting the shape of the output image, given
        by ``(N1, N2)``
    Returns
    -------
    projection : shape `(N1, N2)`
        Projection of volume onto imaging plane,
        which is taken to be over axis 2.
    """
    N1, N2 = shape[0], shape[1]
    # Round coordinates for binning
    rounded_coords = jnp.rint(coords).astype(int)
    # Shift coordinates back to zero in the corner, rather than center
    x_coords, y_coords = (
        rounded_coords[:, 0] + N1 // 2,
        rounded_coords[:, 1] + N2 // 2,
    )
    # Bin values on the same y-z plane
    flat_coords = jnp.ravel_multi_index(
        (x_coords, y_coords), (N1, N2), mode="clip"
    )
    projection = jnp.bincount(
        flat_coords, weights=density, length=N1 * N2
    ).reshape((N1, N2))

    return projection
