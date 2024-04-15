import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._config import ImageConfig
from .._potential import RealVoxelGridPotential, RealVoxelGridPotentialInterpolator
from .potential_integrator import AbstractPotentialIntegrator


class MultiSlice(AbstractPotentialIntegrator, strict=True):
    def __call__(
        self,
        potential: RealVoxelGridPotential | RealVoxelGridPotentialInterpolator,
        wavelength_in_angstroms: Float,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        real_slices = potential.real_voxel_grid.swapaxes(0, 2).swapaxes(
            0, 1
        )  # z-axis is the first axis
        shape = real_slices.shape[1:]
        plane_wave_n = jnp.ones(shape, dtype=jnp.complex64)

        sigma = 1
        transmission = jnp.exp(1j * sigma * real_slices)
        delta_z = config.voxel_size
        # voxel_size = 1
        # config = cxs.ImageConfig(shape=shape, pixel_size=voxel_size)
        kx, ky = config.wrapped_coordinate_grid_in_angstroms.array.T
        k2 = jnp.hypot(kx, ky) ** 2
        # wavelength_in_angstroms = 0.001
        fresnel_propagator = jnp.exp(
            -1j * jnp.pi * wavelength_in_angstroms * k2 * delta_z
        )

        plane_wave_ns = jnp.zeros((len(real_slices), *shape), dtype=jnp.complex64)
        for n in range(len(transmission)):
            tn = transmission[n]
            t_psi_f = jnp.fft.fftn(tn * plane_wave_n)
            t_psi_f_shift = jnp.fft.fftshift(t_psi_f)
            plane_wave_ns = plane_wave_ns.at[n].set(
                jnp.fft.ifftn(t_psi_f_shift * fresnel_propagator)
            )

        return plane_wave_ns[-1]
