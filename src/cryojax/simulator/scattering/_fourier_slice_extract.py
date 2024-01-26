"""
Scattering methods for the fourier slice theorem.
"""

from __future__ import annotations

__all__ = [
    "extract_slice",
    "extract_slice_with_cubic_spline",
    "FourierSliceExtract",
]

from typing import Any

import jax.numpy as jnp

from ._scattering_method import AbstractProjectionMethod
from ..density import FourierVoxelGrid, FourierVoxelGridAsSpline
from ...image import map_coordinates, map_coordinates_with_cubic_spline
from ...core import field
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
        if isinstance(density, FourierVoxelGridAsSpline):
            return extract_slice_with_cubic_spline(
                density.spline_coefficients,
                frequency_slice,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        else:
            return extract_slice(
                density.fourier_density_grid,
                frequency_slice,
                interpolation_order=self.interpolation_order,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
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
    logical_frequency_slice = _convert_frequencies_to_indices(frequency_slice)
    # Only take the lower half plane
    logical_frequency_slice = _extract_lower_half_plane(
        logical_frequency_slice
    )
    # Convert arguments to map_coordinates convention and compute
    k_x, k_y, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates(
        fourier_density_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
    )[:, :, 0]

    return jnp.fft.ifftshift(projection, axes=(0,))


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
    logical_frequency_slice = _convert_frequencies_to_indices(frequency_slice)
    # Only take the lower half plane
    logical_frequency_slice = _extract_lower_half_plane(
        logical_frequency_slice
    )
    # Convert arguments to map_coordinates convention and compute
    k_x, k_y, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates_with_cubic_spline(
        spline_coefficients, (k_x, k_y, k_z), **kwargs
    )[:, :, 0]

    return jnp.fft.ifftshift(projection, axes=(0,))


def _convert_frequencies_to_indices(frequency_slice: VolumeSliceCoords):
    """Convert a frequency coordinate system with the zero frequency
    component in the center to logical coordinates.

    Assume the grid is that the slice corresponds to is cubic.
    """
    N = frequency_slice.shape[0]
    grid_shape = jnp.asarray([N, N, N], dtype=float)
    return (frequency_slice * grid_shape) + grid_shape // 2


def _extract_lower_half_plane(frequency_slice: VolumeSliceCoords):
    """Extract the lower half plane of the frequency slice.

    Assume the grid is that the slice corresponds to is cubic.
    """
    N = frequency_slice.shape[0]
    return jnp.flip(frequency_slice[:, : N // 2 + 1], axis=1)
