"""
Using the fourier slice theorem for computing volume projections.
"""

from typing import Any
from equinox import field

import jax.numpy as jnp

from ._potential_integrator import AbstractPotentialIntegrator
from .._config import ImageConfig
from .._potential import FourierVoxelGrid, FourierVoxelGridInterpolator
from ...image import (
    irfftn,
    rfftn,
    map_coordinates,
    map_coordinates_with_cubic_spline,
)
from ...typing import ComplexImage, ComplexCubicVolume, VolumeSliceCoords


class FourierSliceExtract(AbstractPotentialIntegrator, strict=True):
    """Integrate points to the exit plane using the
    Fourier-projection slice theorem.

    This extracts slices using resampling techniques housed in
    ``cryojax.image._map_coordinates``. See here for more documentation.

    Attributes
    ----------
    interpolation_order :
        The interpolation order. This can be ``0`` (nearest-neighbor), ``1``
        (linear), or ``3`` (cubic).
        Note that this argument is ignored if a ``FourierVoxelGridInterpolator``
        is passed.
    interpolation_mode :
        Specify how to handle out of bounds indexing.
    interpolation_cval :
        Value for filling out-of-bounds indices. Used only when
        ``interpolation_mode = "fill"``.
    """

    config: ImageConfig

    interpolation_order: int = field(static=True, default=1)
    interpolation_mode: str = field(static=True, default="fill")
    interpolation_cval: complex = field(static=True, default=0.0 + 0.0j)

    def integrate_potential(
        self, potential: FourierVoxelGrid | FourierVoxelGridInterpolator
    ) -> ComplexImage:
        """Compute a projection of the real-space potential by extracting
        a central slice in fourier space.
        """
        frequency_slice = potential.frequency_slice.get()
        N = frequency_slice.shape[0]
        if potential.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(potential, FourierVoxelGridInterpolator):
            fourier_projection = extract_slice_with_cubic_spline(
                potential.coefficients,
                frequency_slice,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        elif isinstance(potential, FourierVoxelGrid):
            fourier_projection = extract_slice(
                potential.fourier_voxel_grid,
                frequency_slice,
                interpolation_order=self.interpolation_order,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        else:
            raise ValueError(
                "Supported density representations are FourierVoxelGrid and FourierVoxelGridInterpolator."
            )

        # Resize the image to match the ImageConfig.padded_shape
        if self.config.padded_shape == (N, N):
            return fourier_projection
        else:
            return rfftn(
                self.config.crop_or_pad_to_padded_shape(
                    irfftn(fourier_projection, s=(N, N))
                )
            )


def extract_slice(
    fourier_voxel_grid: ComplexCubicVolume,
    frequency_slice: VolumeSliceCoords,
    interpolation_order: int = 1,
    **kwargs: Any,
) -> ComplexImage:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    fourier_voxel_grid : shape `(N, N, N)`
        Density grid in fourier space. The zero frequency component
        should be in the center.
    frequency_slice : shape `(N, N, 1, 3)`
        Frequency central slice coordinate system, with the zero
        frequency component in the corner.
    interpolation_order : int
        Order of interpolation, either 0, 1, or 3.
    kwargs
        Keyword arguments passed to ``cryojax.image.map_coordinates``
        or ``cryojax.image.map_coordinates_with_cubic_spline``.

    Returns
    -------
    projection : shape `(N, N//2+1)`
        The output image in fourier space.
    """
    # Convert to logical coordinates
    N = frequency_slice.shape[0]
    logical_frequency_slice = (frequency_slice * N) + N // 2
    # Convert arguments to map_coordinates convention and compute
    k_x, k_y, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates(
        fourier_voxel_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
    )[:, :, 0]
    # Shift zero frequency component to corner and take upper half plane
    projection = jnp.fft.ifftshift(projection)[:, : N // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    if N % 2 == 0:
        projection = projection.at[:, -1].set(0.0 + 0.0j).at[N // 2, :].set(0.0 + 0.0j)
    return projection


def extract_slice_with_cubic_spline(
    spline_coefficients: ComplexCubicVolume,
    frequency_slice: VolumeSliceCoords,
    **kwargs: Any,
) -> ComplexImage:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem, using cubic
    spline coefficients as input.

    Arguments
    ---------
    spline_coefficients : shape `(N+2, N+2, N+2)`
        Coefficients for cubic spline.
    frequency_slice : shape `(N, N, 1, 3)`
        Frequency central slice coordinate system, with the zero
        frequency component in the corner.
    kwargs
        Keyword arguments passed to ``cryojax.image.map_coordinates_with_cubic_spline``.

    Returns
    -------
    projection : shape `(N, N//2+1)`
        The output image in fourier space.
    """
    # Convert to logical coordinates
    N = frequency_slice.shape[0]
    logical_frequency_slice = (frequency_slice * N) + N // 2
    # Convert arguments to map_coordinates convention and compute
    k_x, k_y, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates_with_cubic_spline(
        spline_coefficients, (k_x, k_y, k_z), **kwargs
    )[:, :, 0]
    # Shift zero frequency component to corner and take upper half plane
    projection = jnp.fft.ifftshift(projection)[:, : N // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    return projection if N % 2 == 1 else projection.at[:, -1].set(0.0 + 0.0j)
