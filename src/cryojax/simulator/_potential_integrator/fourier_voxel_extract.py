"""
Using the fourier slice theorem for computing volume projections.
"""

from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...image import (
    fftn,
    ifftn,
    irfftn,
    map_coordinates,
    map_coordinates_with_cubic_spline,
    rfftn,
)
from .._instrument_config import InstrumentConfig
from .._potential_representation import (
    FourierVoxelGridPotential,
    FourierVoxelGridPotentialInterpolator,
)
from .base_potential_integrator import AbstractVoxelPotentialIntegrator


class FourierSliceExtraction(AbstractVoxelPotentialIntegrator, strict=True):
    """Integrate points to the exit plane using the Fourier projection-slice theorem.

    This extracts slices using interpolation methods housed in
    `cryojax.image.map_coordinates` and `cryojax.image.map_coordinates_with_cubic_spline`.
    """

    pixel_rescaling_method: Optional[str]
    interpolation_order: int
    interpolation_mode: str
    interpolation_cval: complex

    def __init__(
        self,
        *,
        pixel_rescaling_method: Optional[str] = None,
        interpolation_order: int = 1,
        interpolation_mode: str = "fill",
        interpolation_cval: complex = 0.0 + 0.0j,
    ):
        """**Arguments:**

        - `pixel_rescaling_method`:
            Method for rescaling the final image to the `InstrumentConfig`
            pixel size. See `cryojax.image.rescale_pixel_size` for documentation.
        - `interpolation_order`:
            The interpolation order. This can be `0` (nearest-neighbor), `1`
            (linear), or `3` (cubic).
            Note that this argument is ignored when using this object with a
            `FourierVoxelGridInterpolator`.
        - `interpolation_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.image.map_coordinates` for documentation.
        - `interpolation_cval`:
            Value for filling out-of-bounds indices. Used only when
            `interpolation_mode = "fill"`.
        """
        self.pixel_rescaling_method = pixel_rescaling_method
        self.interpolation_order = interpolation_order
        self.interpolation_mode = interpolation_mode
        self.interpolation_cval = interpolation_cval

    @override
    def compute_fourier_integrated_potential(
        self,
        potential: FourierVoxelGridPotential | FourierVoxelGridPotentialInterpolator,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the integrated scattering potential at the `InstrumentConfig` settings
        of a voxel-based representation in fourier-space, using fourier slice extraction.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `instrument_config`: The configuration of the resulting image.

        **Returns:**

        The extracted fourier voxels of the `potential`, at the
        `instrument_config.padded_shape` and the `instrument_config.pixel_size`.
        """
        frequency_slice = potential.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        if potential.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(potential, FourierVoxelGridPotentialInterpolator):
            fourier_projection = self.extract_fourier_slice_from_spline_coefficients(
                potential.coefficients,
                frequency_slice,
            )
        elif isinstance(potential, FourierVoxelGridPotential):
            fourier_projection = self.extract_fourier_slice_from_grid_points(
                potential.fourier_voxel_grid,
                frequency_slice,
            )
        else:
            raise ValueError(
                "Supported types for `potential` are `FourierVoxelGridPotential` and "
                "`FourierVoxelGridPotentialInterpolator`."
            )

        # Resize the image to match the InstrumentConfig.padded_shape
        if instrument_config.padded_shape != (N, N):
            fourier_projection = rfftn(
                instrument_config.crop_or_pad_to_padded_shape(
                    irfftn(fourier_projection, s=(N, N))
                )
            )
        return self._convert_raw_image_to_integrated_potential(
            fourier_projection, potential, instrument_config, is_hermitian_symmetric=True
        )

    def extract_fourier_slice_from_spline_coefficients(
        self,
        spline_coefficients: Complex[Array, "dim+2 dim+2 dim+2"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
    ) -> Complex[Array, "dim dim//2+1"]:
        """Extract a fourier slice using the interpolation defined by
        `spline_coefficients` at coordinates `frequency_slice_in_pixels`.

        **Arguments:**

        - `spline_coefficients`:
            Spline coefficients of the density grid in fourier space.
            The coefficients should be computed from a `fourier_voxel_grid`
            with the zero frequency component in the center. These are
            typically computed with the function
            `cryojax.image.compute_spline_coefficients`.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`. This argument is
            not used in the `FourierSliceExtraction` class.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam. This argument is
            not used in the `FourierSliceExtraction` class.

        **Returns:**

        The interpolated fourier slice at coordinates `frequency_slice_in_pixels`.
        """
        return _extract_slice_with_cubic_spline(
            spline_coefficients,
            frequency_slice_in_pixels,
            mode=self.interpolation_mode,
            cval=self.interpolation_cval,
        )

    def extract_fourier_slice_from_grid_points(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
    ) -> Complex[Array, "dim dim//2+1"]:
        """Extract a fourier slice of the `fourier_voxel_grid` at coordinates
        `frequency_slice_in_pixels`.

        **Arguments:**

        - `fourier_voxel_grid`:
            Density grid in fourier space. The zero frequency component
            should be in the center.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`. This argument is
            not used in the `FourierSliceExtraction` class.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam. This argument is
            not used in the `FourierSliceExtraction` class.

        **Returns:**

        The interpolated fourier slice at coordinates `frequency_slice_in_pixels`.
        """
        return _extract_slice(
            fourier_voxel_grid,
            frequency_slice_in_pixels,
            interpolation_order=self.interpolation_order,
            mode=self.interpolation_mode,
            cval=self.interpolation_cval,
        )


class EwaldSphereExtraction(AbstractVoxelPotentialIntegrator, strict=True):
    """Integrate points to the exit plane by extracting a surface of the ewald
    sphere in fourier space.

    This extracts surfaces using interpolation methods housed in
    `cryojax.image.map_coordinates` and `cryojax.image.map_coordinates_with_cubic_spline`.
    """

    pixel_rescaling_method: Optional[str]
    interpolation_order: int
    interpolation_mode: str
    interpolation_cval: complex

    def __init__(
        self,
        *,
        pixel_rescaling_method: Optional[str] = None,
        interpolation_order: int = 1,
        interpolation_mode: str = "fill",
        interpolation_cval: complex = 0.0 + 0.0j,
    ):
        """**Arguments:**

        - `pixel_rescaling_method`:
            Method for rescaling the final image to the `InstrumentConfig`
            pixel size. See `cryojax.image.rescale_pixel_size` for documentation.
        - `interpolation_order`:
            The interpolation order. This can be `0` (nearest-neighbor), `1`
            (linear), or `3` (cubic).
            Note that this argument is ignored when using this object with a
            `FourierVoxelGridInterpolator`.
        - `interpolation_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.image.map_coordinates` for documentation.
        - `interpolation_cval`:
            Value for filling out-of-bounds indices. Used only when
            `interpolation_mode = "fill"`.
        """
        self.pixel_rescaling_method = pixel_rescaling_method
        self.interpolation_order = interpolation_order
        self.interpolation_mode = interpolation_mode
        self.interpolation_cval = interpolation_cval

    @override
    def compute_fourier_integrated_potential(
        self,
        potential: FourierVoxelGridPotential | FourierVoxelGridPotentialInterpolator,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        """Compute the integrated scattering potential at the `InstrumentConfig` settings
        of a voxel-based representation in fourier-space, using fourier slice extraction.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `instrument_config`: The configuration of the resulting image.

        **Returns:**

        The extracted fourier voxels of the `potential`, at the
        `instrument_config.padded_shape` and the `instrument_config.pixel_size`.
        """
        frequency_slice = potential.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        if potential.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(potential, FourierVoxelGridPotentialInterpolator):
            ewald_sphere_surface = self.extract_ewald_sphere_from_spline_coefficients(
                potential.coefficients,
                frequency_slice,
                potential.voxel_size,
                instrument_config.wavelength_in_angstroms,
            )
        elif isinstance(potential, FourierVoxelGridPotential):
            ewald_sphere_surface = self.extract_ewald_sphere_from_grid_points(
                potential.fourier_voxel_grid,
                frequency_slice,
                potential.voxel_size,
                instrument_config.wavelength_in_angstroms,
            )
        else:
            raise ValueError(
                "Supported types for `potential` are `FourierVoxelGridPotential` and "
                "`FourierVoxelGridPotentialInterpolator`."
            )

        # Resize the image to match the InstrumentConfig.padded_shape
        if instrument_config.padded_shape != (N, N):
            ewald_sphere_surface = fftn(
                instrument_config.crop_or_pad_to_padded_shape(
                    ifftn(ewald_sphere_surface, s=(N, N))
                )
            )
        return self._convert_raw_image_to_integrated_potential(
            ewald_sphere_surface,
            potential,
            instrument_config,
            is_hermitian_symmetric=False,
        )

    def extract_ewald_sphere_from_spline_coefficients(
        self,
        spline_coefficients: Complex[Array, "dim+2 dim+2 dim+2"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
        voxel_size: Float[Array, ""],
        wavelength_in_angstroms: Float[Array, ""],
    ) -> Complex[Array, "dim dim"]:
        """Extract an ewald sphere surface of the `fourier_voxel_grid` at
        coordinates normal to `frequency_slice_in_pixels` at wavelength
        `wavelength_in_angstroms`.

        **Arguments:**

        - `spline_coefficients`:
            Spline coefficients of the density grid in fourier space.
            The coefficients should be computed from a `fourier_voxel_grid`
            with the zero frequency component in the center. These are
            typically computed with the function
            `cryojax.image.compute_spline_coefficients`.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam.

        **Returns:**

        The interpolated ewald sphere surface at coordinates normal to
        `frequency_slice_in_pixels`.
        """
        return _extract_ewald_sphere_surface_with_cubic_spline(
            spline_coefficients,
            frequency_slice_in_pixels,
            voxel_size,
            wavelength_in_angstroms,
            mode=self.interpolation_mode,
            cval=self.interpolation_cval,
        )

    def extract_ewald_sphere_from_grid_points(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
        voxel_size: Float[Array, ""],
        wavelength_in_angstroms: Float[Array, ""],
    ) -> Complex[Array, "dim dim"]:
        """Extract an ewald sphere surface of the `fourier_voxel_grid` at
        coordinates normal to `frequency_slice_in_pixels` at wavelength
        `wavelength_in_angstroms`.

        **Arguments:**

        - `fourier_voxel_grid`:
            Density grid in fourier space. The zero frequency component
            should be in the center.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam.

        **Returns:**

        The interpolated ewald sphere surface at coordinates normal to
        `frequency_slice_in_pixels`.
        """
        return _extract_ewald_sphere_surface(
            fourier_voxel_grid,
            frequency_slice_in_pixels,
            voxel_size,
            wavelength_in_angstroms,
            interpolation_order=self.interpolation_order,
            mode=self.interpolation_mode,
            cval=self.interpolation_cval,
        )


def _extract_slice(
    fourier_voxel_grid,
    frequency_slice,
    interpolation_order,
    **kwargs,
) -> Complex[Array, "dim dim//2+1"]:
    return _apply_hermitian_symmetry_to_fourier_slice(
        _extract_surface_from_voxel_grid(
            fourier_voxel_grid,
            frequency_slice,
            is_spline_coefficients=False,
            interpolation_order=interpolation_order,
            **kwargs,
        )
    )


def _extract_slice_with_cubic_spline(
    spline_coefficients, frequency_slice, **kwargs
) -> Complex[Array, "dim dim//2+1"]:
    return _apply_hermitian_symmetry_to_fourier_slice(
        _extract_surface_from_voxel_grid(
            spline_coefficients, frequency_slice, is_spline_coefficients=True, **kwargs
        )
    )


def _extract_ewald_sphere_surface(
    fourier_voxel_grid,
    frequency_slice,
    voxel_size,
    wavelength,
    interpolation_order,
    **kwargs,
) -> Complex[Array, "dim dim"]:
    ewald_sphere_frequencies = _get_ewald_sphere_surface_from_slice(
        frequency_slice, voxel_size, wavelength
    )
    return _extract_surface_from_voxel_grid(
        fourier_voxel_grid,
        ewald_sphere_frequencies,
        is_spline_coefficients=False,
        interpolation_order=interpolation_order,
        **kwargs,
    )


def _extract_ewald_sphere_surface_with_cubic_spline(
    spline_coefficients, frequency_slice, voxel_size, wavelength, **kwargs
) -> Complex[Array, "dim dim"]:
    ewald_sphere_frequencies = _get_ewald_sphere_surface_from_slice(
        frequency_slice, voxel_size, wavelength
    )
    return _extract_surface_from_voxel_grid(
        spline_coefficients,
        ewald_sphere_frequencies,
        is_spline_coefficients=True,
        **kwargs,
    )


def _get_ewald_sphere_surface_from_slice(
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
    voxel_size: Float[Array, ""],
    wavelength: Float[Array, ""],
) -> Float[Array, "1 dim dim 3"]:
    frequency_slice_with_zero_in_corner = jnp.fft.ifftshift(
        frequency_slice_in_pixels, axes=(0, 1, 2)
    )
    # Get zhat unit vector of the frequency slice
    xhat, yhat = (
        frequency_slice_with_zero_in_corner[0, 0, 1, :],
        frequency_slice_with_zero_in_corner[0, 1, 0, :],
    )
    xhat, yhat = xhat / jnp.linalg.norm(xhat), yhat / jnp.linalg.norm(yhat)
    zhat = jnp.cross(xhat, yhat)
    # Compute the ewald sphere surface, assuming the frequency slice is
    # in a rotated frame
    q_at_slice = frequency_slice_in_pixels
    q_squared = jnp.sum(q_at_slice**2, axis=-1)
    q_at_surface = (
        q_at_slice
        + (wavelength / voxel_size)
        * (q_squared[..., None] * zhat[None, None, None, :])
        / 2
    )
    return q_at_surface


def _extract_surface_from_voxel_grid(
    voxel_grid,
    frequency_coordinates,
    is_spline_coefficients=False,
    interpolation_order=1,
    **kwargs,
):
    # Convert to logical coordinates
    N = frequency_coordinates.shape[1]
    logical_frequency_coordinates = (frequency_coordinates * N) + N // 2
    # Convert arguments to map_coordinates convention and compute
    k_z, k_y, k_x = jnp.transpose(logical_frequency_coordinates, axes=[3, 0, 1, 2])
    if is_spline_coefficients:
        spline_coefficients = voxel_grid
        surface = map_coordinates_with_cubic_spline(
            spline_coefficients, (k_x, k_y, k_z), **kwargs
        )[0, :, :]
    else:
        fourier_voxel_grid = voxel_grid
        surface = map_coordinates(
            fourier_voxel_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
        )[0, :, :]
    # Shift zero frequency component to corner
    surface = jnp.fft.ifftshift(surface)

    return surface


def _apply_hermitian_symmetry_to_fourier_slice(fourier_slice):
    dim = fourier_slice.shape[0]
    # Take upper half plane
    fourier_slice = fourier_slice[:, : dim // 2 + 1]
    # Set last line of frequencies to zero if image dimension is even
    if dim % 2 == 0:
        fourier_slice = (
            fourier_slice.at[:, -1].set(0.0 + 0.0j).at[dim // 2, :].set(0.0 + 0.0j)
        )
    return fourier_slice
