"""
Scattering methods for the fourier slice theorem.
"""

from typing import Any
from equinox import field

import jax.numpy as jnp

from ._scattering_method import AbstractProjectionMethod
from .._density import FourierVoxelGrid, FourierVoxelGridAsSpline
from ...image import (
    irfftn,
    rfftn,
    map_coordinates,
    map_coordinates_with_cubic_spline,
)
from ...typing import ComplexImage, ComplexCubicVolume, VolumeSliceCoords


class FourierSliceExtract(AbstractProjectionMethod):
    """
    Scatter points to the image plane using the
    Fourier-projection slice theorem.

    This extracts slices using resampling techniques housed in
    ``cryojax.image._map_coordinates``. See here for more documentation.

    Attributes
    ----------
    interpolation_order :
        The interpolation order. This can be ``0`` (nearest-neighbor), ``1``
        (linear), or ``3`` (cubic).
        Note that this argument is ignored if a ``FourierVoxelGridAsSpline``
        is passed.
    interpolation_mode :
        Specify how to handle out of bounds indexing.
    interpolation_cval :
        Value for filling out-of-bounds indices. Used only when
        ``interpolation_mode = "fill"``.
    """

    interpolation_order: int = field(static=True, default=1)
    interpolation_mode: str = field(static=True, default="fill")
    interpolation_cval: complex = field(static=True, default=0.0 + 0.0j)

    def project_density(self, density: FourierVoxelGrid) -> ComplexImage:
        """
        Compute an image by sampling a slice in the
        rotated fourier transform and interpolating onto
        a uniform grid in the object plane.
        """
        frequency_slice = density.frequency_slice.get()
        N = frequency_slice.shape[0]
        if density.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(density, FourierVoxelGridAsSpline):
            fourier_projection = extract_slice_with_cubic_spline(
                density.spline_coefficients,
                frequency_slice,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        elif isinstance(density, FourierVoxelGrid):
            fourier_projection = extract_slice(
                density.fourier_density_grid,
                frequency_slice,
                interpolation_order=self.interpolation_order,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        else:
            raise ValueError(
                "Supported density representations are FourierVoxelGrid and FourierVoxelGridAsSpline."
            )
        # Resize the image to match the ImageManager.padded_shape
        if self.manager.padded_shape == (N, N):
            return fourier_projection
        else:
            return rfftn(
                self.manager.crop_or_pad_to_padded_shape(
                    irfftn(fourier_projection, s=(N, N))
                )
            )


def extract_slice(
    fourier_density_grid: ComplexCubicVolume,
    frequency_slice: VolumeSliceCoords,
    interpolation_order: int = 1,
    **kwargs: Any,
) -> ComplexImage:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    fourier_density_grid : shape `(N, N, N)`
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
    k_y, k_x, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates(
        fourier_density_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
    )[:, :, 0]
    # Shift zero frequency component to corner and take upper half plane
    projection = jnp.fft.ifftshift(projection)[:, : N // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    return projection if N % 2 == 1 else projection.at[:, -1].set(0.0 + 0.0j)


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
    k_y, k_x, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates_with_cubic_spline(
        spline_coefficients, (k_x, k_y, k_z), **kwargs
    )[:, :, 0]
    # Shift zero frequency component to corner and take upper half plane
    projection = jnp.fft.ifftshift(projection)[:, : N // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    return projection if N % 2 == 1 else projection.at[:, -1].set(0.0 + 0.0j)
