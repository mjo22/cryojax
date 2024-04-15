import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Complex, Float

from .._config import ImageConfig
from .._potential import RealVoxelGridPotential
from .potential_integrator import AbstractPotentialIntegrator


class MultiSliceIntegrator(AbstractPotentialIntegrator, strict=True):
    delta_z: int = field(static=True, default=1)  # pixels
    sigma: int = field(static=True, default=1)  # TODO: get from config

    def __call__(
        self,
        potential: RealVoxelGridPotential,
        wavelength_in_angstroms: Float[Array, ""],
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]:
        real_slices = potential.real_voxel_grid
        shape = real_slices.shape[1:]
        real_slices = real_slices.reshape(-1, self.delta_z, *shape).sum(1)

        plane_wave_n = jnp.ones(shape, dtype=complex)

        transmission = jnp.exp(1j * self.sigma * real_slices)
        # delta_z = config.voxel_size
        # voxel_size = 1
        # config = cxs.ImageConfig(shape=shape, pixel_size=voxel_size)
        kx, ky = (
            config.wrapped_frequency_grid_in_angstroms.array.T
        )  # TODO: not centered. see _coordinate_functions.make_frequencies
        k2 = jnp.hypot(kx, ky) ** 2
        # wavelength_in_angstroms = 0.001
        fresnel_propagator = jnp.exp(
            -1j
            * jnp.pi
            * wavelength_in_angstroms
            * k2
            * self.delta_z
            * config.voxel_size
        )

        plane_wave_ns = jnp.zeros((len(real_slices), *shape), dtype=complex)
        for n in range(len(transmission)):
            tn = transmission[n]
            t_psi_f = jnp.fft.fftn(tn * plane_wave_n)
            # t_psi_f_shift = jnp.fft.fftshift(t_psi_f)
            plane_wave_ns = plane_wave_ns.at[n].set(
                jnp.fft.ifftn(
                    t_psi_f * fresnel_propagator
                )  # TODO: skip last one (move to top of loop)
            )

        exit_wave = plane_wave_ns[-1]  # TODO: return fourier space
        return exit_wave
