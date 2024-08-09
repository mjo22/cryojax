"""Multislice integrator that steps in successive convolutions."""

from typing import Optional
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex

from cryojax.coordinates import make_frequency_grid
from cryojax.image import fftn, ifftn

from .._instrument_config import InstrumentConfig
from .._potential_representation import RealVoxelGridPotential
from .._scattering_theory import compute_phase_shifts_from_integrated_potential
from .base_multislice_integrator import AbstractMultisliceIntegrator


class FFTMultisliceIntegrator(
    AbstractMultisliceIntegrator[RealVoxelGridPotential],
    strict=True,
):
    n_slices_per_step: int
    pixel_rescaling_method: Optional[str]

    def __init__(
        self,
        *,
        n_slices_per_step: int = 1,
        pixel_rescaling_method: Optional[str] = None,
    ):
        """**Arguments:**

        - `n_slices_per_step`:
            The number of slices to step through per iteration.
        - `pixel_rescaling_method`:
            Method for rescaling the final image to the `InstrumentConfig`
            pixel size. See `cryojax.image.rescale_pixel_size` for documentation.
        """
        self.n_slices_per_step = n_slices_per_step
        self.pixel_rescaling_method = pixel_rescaling_method

    @override
    def compute_wavefunction_at_exit_plane(
        self,
        potential: RealVoxelGridPotential,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Initialize geometry
        shape_z, shape_xy = potential.shape[0], potential.shape[1:]
        n_slices = shape_z
        slice_thickness = (
            potential.voxel_size
        )  # * self.n_slices_per_step (not implemented)
        # Compute the integrated potential in a given slice interval, multiplying by
        # the slice thickness (TODO: interpolate for different slice thicknesses?
        # compute for the rotated coordinate system stored in the potential?)
        integrated_potential_per_slice = potential.real_voxel_grid * slice_thickness
        phase_shifts_per_slice = jax.vmap(
            compute_phase_shifts_from_integrated_potential, in_axes=[0, None]
        )(integrated_potential_per_slice, instrument_config.wavelength_in_angstroms)
        # Compute the transmission function
        transmission = jnp.exp(1.0j * phase_shifts_per_slice)
        # Compute the fresnel propagator (TODO: check numerical factors)
        frequency_grid = make_frequency_grid(
            shape_xy, potential.voxel_size, half_space=False
        )  # not ideal to make a new grid every slice?
        radial_frequency_grid = jnp.sum(frequency_grid**2, axis=-1)
        fresnel_propagator = jnp.exp(
            1.0j
            * jnp.pi
            * instrument_config.wavelength_in_angstroms
            * radial_frequency_grid
            * slice_thickness
        )
        # Prepare for iteration. First, initialize plane wave
        plane_wave = jnp.ones(shape_xy, dtype=complex)
        # ... stepping function
        make_step = lambda n, last_exit_wave: ifftn(
            fftn(transmission[n, :, :] * last_exit_wave) * fresnel_propagator
        )
        # Compute exit wave
        exit_wave = jax.lax.fori_loop(0, n_slices, make_step, plane_wave)

        return self._postprocess_exit_wave(exit_wave, potential, instrument_config)
