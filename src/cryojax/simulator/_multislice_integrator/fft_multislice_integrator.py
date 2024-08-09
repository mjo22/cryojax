from typing import Any
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex

from cryojax.image import fftn, ifftn

from .._instrument_config import InstrumentConfig
from .._potential_representation import AbstractAtomicPotential
from .._scattering_theory import compute_phase_shifts_from_integrated_potential
from .base_multislice_integrator import AbstractMultisliceIntegrator


class FFTMultisliceIntegrator(
    AbstractMultisliceIntegrator[AbstractAtomicPotential],
    strict=True,
):
    """Multislice integrator that steps using successive FFT-based convolutions."""

    slice_thickness_in_voxels: int
    options_for_rasterization: dict[str, Any]

    def __init__(
        self,
        slice_thickness_in_voxels: int = 1,
        *,
        options_for_rasterization: dict[str, Any],
    ):
        """**Arguments:**

        - `slice_thickness_in_voxels`:
            The number of slices to step through per iteration of the
            rasterized voxel grid.
        - `options_for_rasterization`:
            See `cryojax.simulator.AbstractAtomicPotential.as_real_voxel_grid`
            for documentation.
        """
        if slice_thickness_in_voxels < 1:
            raise AttributeError(
                "FFTMultisliceIntegrator.slice_thickness_in_voxels must be an "
                "integer greater than or equal to 1."
            )
        self.slice_thickness_in_voxels = slice_thickness_in_voxels
        self.options_for_rasterization = options_for_rasterization

    @override
    def compute_wavefunction_at_exit_plane(
        self,
        potential: AbstractAtomicPotential,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Rasterize 3D potential
        dim = min(instrument_config.padded_shape)
        pixel_size = instrument_config.pixel_size
        potential_voxel_grid = potential.as_real_voxel_grid(
            (dim, dim, dim), pixel_size, **self.options_for_rasterization
        )
        # Initialize multislice geometry
        shape_xy = (dim, dim)
        n_slices = dim // self.slice_thickness_in_voxels
        slice_thickness = pixel_size * self.slice_thickness_in_voxels
        # Locally average the potential to be at the given slice thickness.
        # Thow away some slices equal to the remainder
        # `dim % self.slice_thickness_in_voxels`
        if self.slice_thickness_in_voxels > 1:
            potential_voxel_grid = jnp.mean(
                potential_voxel_grid[
                    : dim - dim % self.slice_thickness_in_voxels, ...
                ].reshape((self.slice_thickness_in_voxels, n_slices, dim, dim)),
                axis=0,
            )
        # Compute the integrated potential in a given slice interval, multiplying by
        # the slice thickness (TODO: interpolate for different slice thicknesses?)
        integrated_potential_per_slice = potential_voxel_grid * slice_thickness
        phase_shifts_per_slice = jax.vmap(
            compute_phase_shifts_from_integrated_potential, in_axes=[0, None]
        )(integrated_potential_per_slice, instrument_config.wavelength_in_angstroms)
        # Compute the transmission function
        transmission = jnp.exp(1.0j * phase_shifts_per_slice)
        # Compute the fresnel propagator (TODO: check numerical factors)
        radial_frequency_grid = jnp.sum(
            instrument_config.padded_full_frequency_grid_in_angstroms**2, axis=-1
        )
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

        return self._postprocess_exit_wave(exit_wave, instrument_config)
