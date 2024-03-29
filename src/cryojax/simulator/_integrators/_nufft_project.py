"""
Using non-uniform FFTs for computing volume projections.
"""

import math
from typing import Union

import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Complex

from ...typing import (
    PointCloudCoords2D,
    PointCloudCoords3D,
    RealNumber,
    RealPointCloud,
)
from .._config import ImageConfig
from .._potential import RealVoxelCloudPotential, RealVoxelGridPotential
from ._potential_integrator import AbstractPotentialIntegrator


class NufftProject(AbstractPotentialIntegrator, strict=True):
    """Integrate points onto the exit plane using
    non-uniform FFTs.

    Attributes
    ----------
    eps : `float`
        See ``jax-finufft`` for documentation.
    """

    eps: float = field(static=True, default=1e-6)

    def __call__(
        self,
        potential: RealVoxelGridPotential | RealVoxelCloudPotential,
        wavelength_in_angstroms: RealNumber,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Rasterize image with non-uniform FFTs."""
        if isinstance(potential, RealVoxelGridPotential):
            shape = potential.shape
            fourier_projection = project_with_nufft(
                potential.real_voxel_grid.ravel(),
                potential.wrapped_coordinate_grid_in_pixels.get().reshape(
                    (math.prod(shape), 3)
                ),
                config.padded_shape,
                eps=self.eps,
            )
        elif isinstance(potential, RealVoxelCloudPotential):
            fourier_projection = project_with_nufft(
                potential.voxel_weights,
                potential.wrapped_coordinate_list_in_pixels.get(),
                config.padded_shape,
                eps=self.eps,
            )
        else:
            raise ValueError(
                "Supported density representations are RealVoxelGrid and VoxelCloud."
            )
        # Rescale the voxel size to the ImageConfig.pixel_size
        return config.rescale_to_pixel_size(
            fourier_projection, potential.voxel_size, is_real=False
        )


def project_with_nufft(
    weights: RealPointCloud,
    coordinate_list: Union[PointCloudCoords2D, PointCloudCoords3D],
    shape: tuple[int, int],
    eps: float = 1e-6,
) -> Complex[Array, "{shape[0]} {shape[1]}"]:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    Arguments
    ---------
    weights : shape `(N,)`
        Density point cloud.
    coordinates : shape `(N, 2)` or shape `(N, 3)`
        Coordinate system of point cloud.
    shape :
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    from jax_finufft import nufft1

    weights, coordinate_list = (
        jnp.asarray(weights).astype(complex),
        jnp.asarray(coordinate_list),
    )
    # Get x and y coordinates
    coordinates_xy = coordinate_list[:, :2]
    # Normalize coordinates betweeen -pi and pi
    M1, M2 = shape
    image_size = jnp.asarray((M1, M2), dtype=float)
    coordinates_periodic = 2 * jnp.pi * coordinates_xy / image_size
    # Unpack and compute
    x, y = coordinates_periodic[:, 0], coordinates_periodic[:, 1]
    projection = nufft1(shape, weights, y, x, eps=eps, iflag=-1)
    # Shift zero frequency component to corner and take upper half plane
    projection = jnp.fft.ifftshift(projection)[:, : M2 // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    if M2 % 2 == 0:
        projection = projection.at[:, -1].set(0.0 + 0.0j)
    if M1 % 2 == 0:
        projection = projection.at[M1 // 2, :].set(0.0 + 0.0j)
    return projection
