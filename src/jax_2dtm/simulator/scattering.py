"""
Routines to model image formation.
"""

__all__ = ["project", "ScatteringImage"]

import dataclasses
import jax.numpy as jnp
from .cloud import Cloud, rotate_and_translate
from .image import ImageConfig, ImageModel
from .state import ParameterState
from ..types import Array, Scalar
from ..utils import nufft


@dataclasses.dataclass
class ScatteringImage(ImageModel):
    """"""

    def render(self, params: ParameterState) -> Array:
        pose = params.pose
        transformed_cloud = rotate_and_translate(self.cloud, pose)
        scattering_image = project(transformed_cloud, self.config)

        return scattering_image

    def sample(self, params: ParameterState) -> Array:
        raise NotImplementedError

    def log_likelihood(
        self, observed: Array, params: ParameterState
    ) -> Scalar:
        raise NotImplementedError


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
    projection :
        The output image in the fourier domain.
    """
    projection = nufft(
        (*config.shape, int(1)), *cloud.iter_meta(), eps=config.eps
    )[:, :, 0]

    return projection


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
