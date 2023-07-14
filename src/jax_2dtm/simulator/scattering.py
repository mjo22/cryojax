"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = ["project_with_nufft", "ImageConfig", "ScatteringConfig"]

import jax.numpy as jnp

from ..types import dataclass, field, Array
from ..utils import nufft
from ..core import Serializable


@dataclass
class ImageConfig(Serializable):
    """
    Configuration for an electron microscopy image.

    Attributes
    ----------
    shape : `tuple[int, int]`
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pixel_size : `float`
        Size of camera pixels, in dimensions of length.
    """

    shape: tuple[int, int] = field(pytree_node=False)
    pixel_size: float = field(pytree_node=False)


@dataclass
class ScatteringConfig(ImageConfig):
    """
    Configuration for an image with a given
    scattering method.

    Attributes
    ----------
    eps : `float`
        See ``jax_2dtm.simulator.project_with_nufft``
        for documentation.
    """

    eps: float = field(pytree_node=False, default=1e-6)

    def project(self, *args):
        """Projection method for image rendering."""
        return project_with_nufft(*args, self.shape, eps=self.eps)


def project_with_nufft(
    density: Array,
    coordinates: Array,
    box_size: Array,
    shape: tuple[int, int],
    eps: float = 1e-6,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``jax_2dtm.utils.nufft`` for more detail.

    Arguments
    ---------
    density :
        Density point cloud.
    coordinates :
        Coordinate system of point cloud.
    box_size :
        Box size of point.
    shape : `tuple[int, int]`
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    eps : `float`
        Desired precision in computing the volume
        projection. See `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.

    Returns
    -------
    projection :
        The output image in the fourier domain.
    """
    projection = nufft(
        (*shape, int(1)), density, coordinates, box_size, eps=eps
    )[:, :, 0]

    return projection


def project_with_binning(
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
