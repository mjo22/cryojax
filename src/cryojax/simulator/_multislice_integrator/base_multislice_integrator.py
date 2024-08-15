from abc import abstractmethod
from typing import Generic, TypeVar
from typing_extensions import override

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Complex

from cryojax.image import fftn, ifftn

from .._instrument_config import InstrumentConfig
from .._potential_representation import (
    RealVoxelGridPotential,
)


PotentialT = TypeVar("PotentialT")


class AbstractMultisliceIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a multi-slice integration scheme."""

    @abstractmethod
    def compute_wavefunction_at_exit_plane(
        self,
        potential: PotentialT,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        raise NotImplementedError


class MultisliceIntegrator(
    AbstractMultisliceIntegrator[RealVoxelGridPotential],
    strict=True,
):
    @override
    def compute_wavefunction_at_exit_plane(
        self,
        potential: RealVoxelGridPotential,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        delta_z = 1.0
        real_slices = potential.real_voxel_grid
        shape = potential.shape[1:]
        # TODO: interpolate for different slice thicknesses
        plane_wave_n = jnp.ones(shape, dtype=complex)
        sigma = (
            instrument_config.wavelength_in_angstroms / (4 * jnp.pi)
        )  # see https://github.com/mjo22/cryojax/blob/multislice-updated/src/cryojax/simulator/_scattering_theory/common_functions.py
        transmission = jnp.exp(1j * sigma * real_slices)
        kx, ky = instrument_config.padded_full_frequency_grid_in_angstroms.T
        k2_origin_corner = jnp.hypot(kx, ky) ** 2  # TODO: shift to origin at centre
        k2_origin_center = jnp.fft.fftshift(k2_origin_corner)

        fresnel_propagator = jnp.exp(
            -1j
            * jnp.pi
            * instrument_config.wavelength_in_angstroms
            * k2_origin_center
            * delta_z
            * potential.voxel_size
        )

        plane_wave_ns = jnp.zeros((len(real_slices), *shape), dtype=complex)
        for n in range(len(transmission)):
            tn = transmission[n]
            t_psi_f = fftn(tn * plane_wave_n)
            # t_psi_f_shift = jnp.fft.fftshift(t_psi_f)
            plane_wave_ns = plane_wave_ns.at[n].set(
                ifftn(
                    t_psi_f * fresnel_propagator
                )  # TODO: skip last one (move to top of loop)
            )
        exit_wave = plane_wave_ns[-1]  # TODO: return fourier space

        return exit_wave
