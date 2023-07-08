"""
Routines to model image formation.
"""

from __future__ import annotations

__all__ = ["project_with_nufft", "ScatteringImage"]

import dataclasses

import jax.numpy as jnp
from typing import TYPE_CHECKING

from .image import ImageConfig, ImageModel
from .filters import Filter, AntiAliasingFilter
from ..types import Array, Scalar
from ..utils import nufft

if TYPE_CHECKING:
    from .state import ParameterState


@dataclasses.dataclass
class ScatteringImage(ImageModel):
    """
    Compute the scattering pattern on the imaging plane.
    """

    def __post_init__(self):
        super().__post_init__()
        self.filters: list[Filter] = [
            AntiAliasingFilter(self.config, self.freqs)
        ]

    def render(self, state: "ParameterState") -> Array:
        # Compute scattering at image plane
        cloud = self.cloud.view(state.pose)
        scattering_image = cloud.project(self.config)
        # Apply filters
        for filter in self.filters:
            scattering_image = filter(scattering_image)

        return scattering_image

    def sample(self, state: "ParameterState") -> Array:
        raise NotImplementedError

    def log_likelihood(
        self, observed: Array, state: "ParameterState"
    ) -> Scalar:
        raise NotImplementedError


def project_with_nufft(
    config: ImageConfig,
    density: Array,
    coordinates: Array,
    box_size: Array,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``jax_2dtm.utils.fft.nufft`` for more detail.

    Arguments
    ---------
    density :
        Density point cloud.
    coordinates :
        Coordinate system of point cloud.
    boxsize :
        Box size of point.
    config :
        Image configuation.

    Returns
    -------
    projection :
        The output image in the fourier domain.
    """
    projection = nufft(
        (*config.shape, int(1)), density, coordinates, box_size, eps=config.eps
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
