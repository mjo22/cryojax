from functools import partial
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...coordinates._make_coordinate_grids import make_coordinate_grid
from ...image import downsample_with_fourier_cropping, rfftn
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
    upsampling_factor: Optional[float | int]

    def __init__(
        self,
        *,
        upsampling_factor: float | int = 1,
    ):
        """**Arguments:**
        `upsampling_factor`: The factor by which to upsample the computation of the images. If `upsampling_factor` is greater than 1, the images will be computed at a higher resolution and then downsampled to the original resolution. This can be useful for reducing aliasing artifacts in the images.
        """  # noqa: E501
        self.upsampling_factor = upsampling_factor

    @override
    def compute_fourier_integrated_potential(
        self,
        potential: GaussianMixtureAtomicPotential | PengAtomicPotential,
        instrument_config: InstrumentConfig,
        batch_size: Optional[int] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute a projection from the atomic potential and transform it to Fourier space

        **Arguments:**
        - `potential`: The atomic potential to project.
        - `instrument_config`: The configuration of the imaging instrument.
        - `batch_size`: The number of atoms to process in each batch. This can be used to reduce memory usage.

        **Returns:**
        The Fourier transform of the integrated potential.
        """

        pixel_size = instrument_config.pixel_size / self.upsampling_factor
        shape = (
            instrument_config.padded_y_dim * self.upsampling_factor,
            instrument_config.padded_x_dim * self.upsampling_factor,
        )

        coordinate_grid_in_angstroms = make_coordinate_grid(
            shape,
            pixel_size,
        )
        if isinstance(potential, PengAtomicPotential):
            if potential.b_factors is None:
                gaussian_widths = potential.scattering_factor_b
            else:
                gaussian_widths = (
                    potential.scattering_factor_b + potential.b_factors[:, None]
                )

            gaussian_amplitudes = potential.scattering_factor_a

        elif isinstance(potential, GaussianMixtureAtomicPotential):
            gaussian_amplitudes = potential.gaussian_strengths
            gaussian_widths = potential.gaussian_widths

        else:
            raise ValueError(
                "Supported types for `potential` are `PengAtomicPotential` and "
                " `GaussianMixtureAtomicPotential`."
            )

        projection = _build_real_space_pixels_from_atoms(
            potential.atom_positions,
            gaussian_amplitudes,
            gaussian_widths,
            coordinate_grid_in_angstroms,
            batch_size=batch_size,
        )

        if self.upsampling_factor > 1:
            projection = downsample_with_fourier_cropping(
                projection, self.upsampling_factor
            )
        # Go to fourier space in cryojax's conventions
        return rfftn(projection)


def _evaluate_2d_real_space_gaussian(
    coordinate_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    atom_position: Float[Array, "3"],
    a: Float[Array, "n_gauss"],
    b: Float[Array, "n_gauss"],
) -> Float[Array, "y_dim x_dim"]:
    """Evaluate a gaussian on a 3D grid.

    **Arguments:**

    - `coordinate_grid`: The coordinate system of the grid.
    - `pos`: The center of the gaussian.
    - `a`: A scale factor.
    - `b`: The scale of the gaussian.

    **Returns:**

    The potential of the gaussian on the grid.
    """

    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)

    b_inverse = 4.0 * jnp.pi / b
    grid_x = coordinate_grid_in_angstroms[0, :, 0]
    grid_y = coordinate_grid_in_angstroms[:, 0, 1]

    gauss_x = (
        jnp.exp(-jnp.pi * b_inverse * ((grid_x - atom_position[0]) ** 2)[None, :])
        * a
        * b_inverse
    )
    gauss_y = jnp.exp(-jnp.pi * b_inverse * ((grid_y - atom_position[1]) ** 2))

    image = 4 * jnp.pi * jnp.matmul(gauss_y.T, gauss_x)

    return image


@eqx.filter_jit
def _build_real_space_pixels_from_atoms(
    atom_positions: Float[Array, "n_atoms 3"],
    ff_a: Float[Array, "n_atoms n_gaussians_per_atom"],
    ff_b: Float[Array, "n_atoms n_gaussians_per_atom"],
    coordinate_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    *,
    batch_size: Optional[int] = None,
) -> Float[Array, "y_dim x_dim"]:
    """
    Build a pixel representation of an atomic model.

    **Arguments**

    - `atom_coords`: The coordinates of the atoms.
    - `ff_a`: Intensity values for each Gaussian in the atom
    - `ff_b` : The inverse scale factors for each Gaussian in the atom
    - `coordinate_grid` : The coordinates of each pixel in the grid.

    **Returns:**

    The pixel representation of the atomic model.
    """
    pixel_grid_buffer = jnp.zeros(coordinate_grid_in_angstroms.shape[:-1])

    # TODO: Look into forcing JAX to do in-place updates
    # Below is a first attempt at this with `donate_argnums`, however
    # equinox.internal.while_loop / equinox.internal.scan could also be
    # options
    @partial(jax.jit, donate_argnums=1)
    def brute_force_body_fun(atom_index, potential):
        return potential + _evaluate_2d_real_space_gaussian(
            coordinate_grid_in_angstroms,
            atom_positions[atom_index],
            ff_a[atom_index],
            ff_b[atom_index],
        )

    @partial(jax.jit, donate_argnums=1)
    def batched_body_fun(iteration_index, potential):
        atom_index_batch = jnp.linspace(
            iteration_index * batch_size,
            (iteration_index + 1) * batch_size - 1,
            batch_size,  # type: ignore
            dtype=int,
        )
        return potential + evaluate_2d_atom_potential_batch(atom_index_batch)

    def evaluate_2d_atom_potential_batch(atom_index_batch):
        vmap_evaluate_2d_atom_potential = jax.vmap(
            _evaluate_2d_real_space_gaussian, in_axes=[None, 0, 0, 0]
        )
        return jnp.sum(
            vmap_evaluate_2d_atom_potential(
                coordinate_grid_in_angstroms,
                jnp.take(atom_positions, atom_index_batch, axis=0),
                jnp.take(ff_a, atom_index_batch, axis=0),
                jnp.take(ff_b, atom_index_batch, axis=0),
            ),
            axis=0,
        )

    # Get the number of atoms
    n_atoms = atom_positions.shape[0]
    # Set the logic for the loop based on the batch size
    if batch_size is None:
        # ... if there is no batch size, loop over all atoms
        n_iterations = n_atoms
        body_fun = brute_force_body_fun
        pixel_grid = jax.lax.fori_loop(0, n_atoms, body_fun, pixel_grid_buffer)
    else:
        # ... if there is a batch size, loop over batches of atoms
        n_iterations = n_atoms // batch_size
        body_fun = batched_body_fun
        pixel_grid = jax.lax.fori_loop(0, n_iterations, body_fun, pixel_grid_buffer)
        # ... and take care of any remaining atoms after the loop
        if n_atoms % batch_size > 0:
            pixel_grid += evaluate_2d_atom_potential_batch(
                jnp.arange(n_atoms - n_atoms % batch_size, n_atoms)
            )

    return pixel_grid
