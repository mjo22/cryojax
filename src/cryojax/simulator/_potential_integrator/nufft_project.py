"""
Using non-uniform FFTs for computing volume projections.
"""

import math
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._instrument_config import InstrumentConfig
from .._potential_representation import RealVoxelCloudPotential, RealVoxelGridPotential
from .base_potential_integrator import AbstractVoxelPotentialIntegrator


class NufftProjection(
    AbstractVoxelPotentialIntegrator[RealVoxelGridPotential | RealVoxelCloudPotential],
    strict=True,
):
    """Integrate points onto the exit plane using
    non-uniform FFTs.
    """

    pixel_rescaling_method: str
    eps: float

    def __init__(self, *, pixel_rescaling_method: str = "bicubic", eps: float = 1e-6):
        """**Arguments:**

        - `pixel_rescaling_method`: Method for interpolating the final image to
                                    the `InstrumentConfig` pixel size. See
                                    `cryojax.image._rescale_pixel_size` for documentation.
        - `eps`: See ``jax-finufft`` for documentation.
        """
        self.pixel_rescaling_method = pixel_rescaling_method
        self.eps = eps

    def project_voxel_cloud_with_nufft(
        self,
        weights: Float[Array, " size"],
        coordinate_list: Float[Array, "size 2"] | Float[Array, "size 3"],
        shape: tuple[int, int],
    ) -> Complex[Array, "{shape[0]} {shape[1]}"]:
        """Project and interpolate 3D volume point cloud
        onto imaging plane using a non-uniform FFT.

        **Arguments:**

        - `weights`:
            Density point cloud.
        - `coordinates`:
            Coordinate system of point cloud.
        - `shape`:
            Shape of the imaging plane in pixels.
            ``width, height = shape[0], shape[1]``
            is the size of the desired imaging plane.

        **Returns:**

        The output image in fourier space.
        """
        return _project_with_nufft(weights, coordinate_list, shape, self.eps)

    @override
    def compute_fourier_integrated_potential(
        self,
        potential: RealVoxelGridPotential | RealVoxelCloudPotential,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the integrated scattering potential at the `InstrumentConfig` settings
        of a voxel-based representation in real-space, using non-uniform FFTs.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `instrument_config`: The configuration of the resulting image.
        """
        if isinstance(potential, RealVoxelGridPotential):
            shape = potential.shape
            fourier_projection = self.project_voxel_cloud_with_nufft(
                potential.real_voxel_grid.ravel(),
                potential.coordinate_grid_in_pixels.reshape((math.prod(shape), 3)),
                instrument_config.padded_shape,
            )
        elif isinstance(potential, RealVoxelCloudPotential):
            fourier_projection = self.project_voxel_cloud_with_nufft(
                potential.voxel_weights,
                potential.coordinate_list_in_pixels,
                instrument_config.padded_shape,
            )
        else:
            raise ValueError(
                "Supported density representations are RealVoxelGrid and VoxelCloud."
            )
        return self._convert_fourier_raw_image_to_integrated_potential(
            fourier_projection, potential, instrument_config
        )


def _project_with_nufft(weights, coordinate_list, shape, eps=1e-6):
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
