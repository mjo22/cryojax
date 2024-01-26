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
from jaxtyping import Float, Array

import jax.numpy as jnp

from ._scattering_method import AbstractProjectionMethod
from ..density import FourierVoxelGrid, FourierVoxelGridAsSpline
from ...image import (
    irfftn,
    rfftn,
    map_coordinates,
    map_coordinates_with_cubic_spline,
)
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
        # Compute the fourier projection
        if isinstance(density, FourierVoxelGridAsSpline):
            fourier_projection = extract_slice_with_cubic_spline(
                density.spline_coefficients,
                frequency_slice,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        else:
            fourier_projection = extract_slice(
                density.fourier_density_grid,
                frequency_slice,
                interpolation_order=self.interpolation_order,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
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
    logical_frequency_slice = _convert_frequencies_to_indices(frequency_slice)
    # Only take the upper half plane
    logical_frequency_slice = _extract_upper_half_plane(
        logical_frequency_slice
    )
    # Convert arguments to map_coordinates convention and compute
    k_y, k_x, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates(
        fourier_density_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
    )[:, :, 0]
    # If the image size is even, pad last axis with zeros
    projection = _pad_highest_frequency_with_zeros_if_even(projection)

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
    # Only take the upper half plane
    logical_frequency_slice = _extract_upper_half_plane(
        logical_frequency_slice
    )
    # Convert arguments to map_coordinates convention and compute
    k_y, k_x, k_z = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates_with_cubic_spline(
        spline_coefficients, (k_x, k_y, k_z), **kwargs
    )[:, :, 0]
    # If the image size is even, pad last axis with zeros
    projection = _pad_highest_frequency_with_zeros_if_even(projection)

    return jnp.fft.ifftshift(projection, axes=(0,))


def _convert_frequencies_to_indices(
    frequency_slice: VolumeSliceCoords,
) -> VolumeSliceCoords:
    """Convert a frequency coordinate system with the zero frequency
    component in the center to logical coordinates.

    Assume the grid is that the slice corresponds to is cubic.
    """
    N = frequency_slice.shape[0]
    grid_shape = jnp.asarray([N, N, N], dtype=float)
    return (frequency_slice * grid_shape) + grid_shape // 2


def _extract_upper_half_plane(
    frequency_slice: VolumeSliceCoords,
) -> Float[Array, "N N//2+N%2 3"]:
    """Extract the lower half plane of the frequency slice.

    Assume the grid is that the slice corresponds to is cubic.
    """
    N = frequency_slice.shape[0]
    return frequency_slice[:, N // 2 :]


def _pad_highest_frequency_with_zeros_if_even(
    fourier_projection: ComplexImage,
) -> ComplexImage:
    """If the image size is zero, pad highest frequency with zeros on final axis."""
    N = fourier_projection.shape[0]
    if N % 2 == 0:
        return jnp.pad(
            fourier_projection,
            ((0, 0), (0, 1)),
            mode="constant",
            constant_values=0.0 + 0.0j,
        )
    else:
        return fourier_projection
