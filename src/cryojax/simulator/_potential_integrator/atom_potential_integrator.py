from typing import Optional
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...coordinates import make_1d_coordinate_grid
from ...image import downsample_to_shape_with_fourier_cropping, rfftn
from .._instrument_config import InstrumentConfig
from .._potential_representation import (
    GaussianMixtureAtomicPotential,
    PengAtomicPotential,
)
from .base_potential_integrator import AbstractPotentialIntegrator


class GaussianMixtureProjection(
    AbstractPotentialIntegrator[GaussianMixtureAtomicPotential | PengAtomicPotential],
    strict=True,
):
    upsampling_factor: Optional[int]

    def __init__(self, *, upsampling_factor: Optional[int] = None):
        """**Arguments:**

        - `upsampling_factor`:
            The factor by which to upsample the computation of the images.
            If `upsampling_factor` is greater than 1, the images will be computed
            at a higher resolution and then downsampled to the original resolution.
            This can be useful for reducing aliasing artifacts in the images.
        """  # noqa: E501
        self.upsampling_factor = upsampling_factor

    def __check_init__(self):
        if self.upsampling_factor is not None and self.upsampling_factor < 1:
            raise AttributeError(
                "`GaussianMixtureProjection.upsampling_factor` must "
                f"be greater than `1`. Got a value of {self.upsampling_factor}."
            )

    @override
    def compute_fourier_integrated_potential(
        self,
        potential: GaussianMixtureAtomicPotential | PengAtomicPotential,
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute a projection from the atomic potential and transform it to Fourier
        space.

        **Arguments:**

        - `potential`: The atomic potential to project.
        - `instrument_config`: The configuration of the imaging instrument.

        **Returns:**

        The Fourier transform of the integrated potential.
        """  # noqa: E501

        if self.upsampling_factor is not None:
            pixel_size = instrument_config.pixel_size / self.upsampling_factor
            shape = (
                instrument_config.padded_y_dim * self.upsampling_factor,
                instrument_config.padded_x_dim * self.upsampling_factor,
            )
        else:
            pixel_size = instrument_config.pixel_size
            shape = instrument_config.padded_shape

        grid_x = make_1d_coordinate_grid(shape[1], pixel_size)
        grid_y = make_1d_coordinate_grid(shape[0], pixel_size)

        if isinstance(potential, PengAtomicPotential):
            if potential.b_factors is None:
                gaussian_widths = potential.scattering_factor_b
            else:
                gaussian_widths = (
                    potential.scattering_factor_b + potential.b_factors[:, None]
                )

            gaussian_amplitudes = potential.scattering_factor_a

        elif isinstance(potential, GaussianMixtureAtomicPotential):
            gaussian_amplitudes = potential.gaussian_amplitudes
            gaussian_widths = potential.gaussian_widths

        else:
            raise ValueError(
                "Supported types for `potential` are `PengAtomicPotential` and "
                " `GaussianMixtureAtomicPotential`."
            )

        projection = _evaluate_2d_real_space_gaussian(
            grid_x, grid_y, potential.atom_positions, gaussian_amplitudes, gaussian_widths
        )

        if self.upsampling_factor is not None:
            fourier_projection = downsample_to_shape_with_fourier_cropping(
                projection,
                downsampled_shape=instrument_config.padded_shape,
                get_real=False,
            )
        else:
            fourier_projection = rfftn(projection)

        return fourier_projection


@jax.jit
def _evaluate_2d_real_space_gaussian(
    grid_x: Float[Array, " x_dim"],
    grid_y: Float[Array, " y_dim"],
    atom_positions: Float[Array, "n_atoms 3"],
    a: Float[Array, "n_atoms n_gaussians_per_atom"],
    b: Float[Array, "n_atoms n_gaussians_per_atom"],
) -> Float[Array, "y_dim x_dim"]:
    """Evaluate a gaussian on a 3D grid.

    **Arguments:**

    - `grid_x`: The x-coordinates of the grid.
    - `grid_y`: The y-coordinates of the grid.
    - `pos`: The center of the gaussian.
    - `a`: A scale factor.
    - `b`: The scale of the gaussian.

    **Returns:**

    The potential of the gaussian on the grid.
    """

    b_inverse = 4.0 * jnp.pi / b

    gauss_x = (
        jnp.exp(
            -jnp.pi
            * b_inverse[None, :, :]
            * ((grid_x[:, None] - atom_positions.T[0, :]) ** 2)[:, :, None]
        )
        * a[None, :, :]
        * b_inverse[None, :, :]
    )
    gauss_y = jnp.exp(
        -jnp.pi
        * b_inverse[None, :, :]
        * ((grid_y[:, None] - atom_positions.T[1, :]) ** 2)[:, :, None]
    )

    gauss_x = jnp.transpose(gauss_x, (2, 1, 0))
    gauss_y = jnp.transpose(gauss_y, (2, 0, 1))

    image = 4 * jnp.pi * jnp.sum(jnp.matmul(gauss_y, gauss_x), axis=0)

    return image
