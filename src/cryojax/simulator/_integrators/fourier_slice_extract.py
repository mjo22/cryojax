"""
Using the fourier slice theorem for computing volume projections.
"""

from typing import Any

import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Complex, Float

from ...image import (
    irfftn,
    map_coordinates,
    map_coordinates_with_cubic_spline,
    rfftn,
)
from .._config import ImageConfig
from .._potential import (
    FourierVoxelGridPotential,
    FourierVoxelGridPotentialInterpolator,
)
from .potential_integrator import AbstractPotentialIntegrator


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

    interpolation_order: int = field(static=True, default=1)
    interpolation_mode: str = field(static=True, default="fill")
    interpolation_cval: complex = field(static=True, default=0.0 + 0.0j)

    def __call__(
        self,
        potential: FourierVoxelGridPotential | FourierVoxelGridPotentialInterpolator,
        wavelength_in_angstroms: Float[Array, ""],
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute a projection of the real-space potential by extracting
        a central slice in fourier space.
        """
        frequency_slice = potential.wrapped_frequency_slice_in_pixels.get()
        N = frequency_slice.shape[1]
        if potential.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(potential, FourierVoxelGridPotentialInterpolator):
            fourier_projection = extract_slice_with_cubic_spline(
                potential.coefficients,
                frequency_slice,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        elif isinstance(potential, FourierVoxelGridPotential):
            fourier_projection = extract_slice(
                potential.fourier_voxel_grid,
                frequency_slice,
                interpolation_order=self.interpolation_order,
                mode=self.interpolation_mode,
                cval=self.interpolation_cval,
            )
        else:
            raise ValueError(
                "Supported density representations are FourierVoxelGrid and "
                "FourierVoxelGridInterpolator."
            )

        # Resize the image to match the ImageConfig.padded_shape
        if config.padded_shape != (N, N):
            fourier_projection = rfftn(
                config.crop_or_pad_to_padded_shape(irfftn(fourier_projection, s=(N, N)))
            )
        # Rescale the voxel size to the ImageConfig.pixel_size
        return config.rescale_to_pixel_size(
            fourier_projection, potential.voxel_size, is_real=False
        )


def extract_slice(
    fourier_voxel_grid: Complex[Array, "dim dim dim"],
    frequency_slice: Float[Array, "1 dim dim 3"],
    interpolation_order: int = 1,
    **kwargs: Any,
) -> Complex[Array, "dim dim//2+1"]:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    fourier_voxel_grid : shape `(N, N, N)`
        Density grid in fourier space. The zero frequency component
        should be in the center.
    frequency_slice : shape `(1, N, N, 3)`
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
    N = frequency_slice.shape[1]
    logical_frequency_slice = (frequency_slice * N) + N // 2
    # Convert arguments to map_coordinates convention and compute
    k_z, k_y, k_x = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates(
        fourier_voxel_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
    )[0, :, :]
    # Shift zero frequency component to corner and take upper half plane
    projection = jnp.fft.ifftshift(projection)[:, : N // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    if N % 2 == 0:
        projection = projection.at[:, -1].set(0.0 + 0.0j).at[N // 2, :].set(0.0 + 0.0j)
    return projection


def extract_slice_with_cubic_spline(
    spline_coefficients: Complex[Array, "dim+2 dim+2 dim+2"],
    frequency_slice: Float[Array, "1 dim dim 3"],
    **kwargs: Any,
) -> Complex[Array, "dim dim//2+1"]:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem, using cubic
    spline coefficients as input.

    Arguments
    ---------
    spline_coefficients : shape `(N+2, N+2, N+2)`
        Coefficients for cubic spline.
    frequency_slice : shape `(1, N, N, 3)`
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
    N = frequency_slice.shape[1]
    logical_frequency_slice = (frequency_slice * N) + N // 2
    # Convert arguments to map_coordinates convention and compute
    k_z, k_y, k_x = jnp.transpose(logical_frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates_with_cubic_spline(
        spline_coefficients, (k_x, k_y, k_z), **kwargs
    )[0, :, :]
    # Shift zero frequency component to corner and take upper half plane
    projection = jnp.fft.ifftshift(projection)[:, : N // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    return projection if N % 2 == 1 else projection.at[:, -1].set(0.0 + 0.0j)
