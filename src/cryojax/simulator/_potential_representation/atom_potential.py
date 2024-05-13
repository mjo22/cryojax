"""
Atomistic representation of the scattering potential.
"""

from abc import abstractmethod
from functools import partial
from typing import Optional
from typing_extensions import override, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..._errors import error_if_negative, error_if_not_positive
from ..._loop import fori_loop_tqdm_decorator
from .._pose import AbstractPose
from .base_potential import AbstractPotentialRepresentation


class AbstractAtomicPotential(AbstractPotentialRepresentation, strict=True):
    """Abstract interface for an atom-based scattering potential representation.

    !!! info
        In, `cryojax`, potentials should be built in units of *inverse length squared*,
        $[L]^{-2}$. This rescaled potential is defined to be

        $$U(\\mathbf{x}) = \\frac{2 m e}{\\hbar^2} V(\\mathbf{x}),$$

        where $V$ is the electrostatic potential energy, $\\mathbf{x}$ is a positional
        coordinate, $m$ is the electron mass, and $e$ is the electron charge.

        For a single atom, this rescaled potential has the advantage (among other reasons)
        that under usual scattering approximations (i.e. the first-born approximation), the
        fourier transform of this quantity is closely related to tabulated electron scattering
        factors. In particular, for a single atom with scattering factor $f^{(e)}(\\mathbf{q})$
        and scattering vector $\\mathbf{q}$, its rescaled potential is equal to

        $$U(\\mathbf{x}) = 32 \\pi \\mathcal{F}^{-1}[f^{(e)}](2 \\mathbf{x}),$$

        where $\\mathcal{F}^{-1}$ is the inverse fourier transform. The inverse fourier
        transform is evaluated at $2\\mathbf{x}$ because $2 \\mathbf{q}$ gives the spatial
        frequency $\\boldsymbol{\\xi}$ in the usual crystallographic fourier transform convention,

        $$\\mathcal{F}[f](\\boldsymbol{\\xi}) = \\int d^3\\mathbf{x} \\ \\exp(2\\pi i \\mathbf{x}\\cdot\\boldsymbol{\\xi}) f(\\mathbf{x}).$$

        **References**:

        - For the definition of the rescaled potential, see
        Chapter 69, Page 2003, Equation 69.6 from *Hawkes, Peter W., and Erwin Kasper.
        Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic Press,
        2022.*
        - To work out the correspondence between the rescaled potential and the electron
        scattering factors, see the supplementary information from *Vulović, Miloš, et al.
        "Image formation modeling in cryo-electron microscopy." Journal of structural
        biology 183.1 (2013): 19-32.*
    """  # noqa: E501

    atom_positions: eqx.AbstractVar[Float[Array, "n_atoms 3"]]

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `atom_positions`."""
        return eqx.tree_at(
            lambda d: d.atom_positions,
            self,
            pose.rotate_coordinates(self.atom_positions),
        )

    @abstractmethod
    def as_real_voxel_grid(
        self,
        coordinate_grid_in_angstroms: Float[Array, "z_dim y_dim x_dim 3"],
        *,
        batch_size: Optional[int] = None,
        progress_bar: bool = False,
        print_every: Optional[int] = None,
    ) -> Float[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError


class GaussianMixtureAtomicPotential(AbstractAtomicPotential, strict=True):
    """An atomistic representation of scattering potential as a mixture of
    gaussians.

    The naming and numerical convention of parameters `atom_strengths` and
    `atom_b_factors` follows "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996), where $a_i$ are
    the `atom_strengths` and $b_i$ are the `atom_b_factors`.
    """

    atom_positions: Float[Array, "n_atoms 3"]
    atom_strengths: Float[Array, "n_atoms n_gaussians_per_atom"]
    atom_b_factors: Float[Array, "n_atoms n_gaussians_per_atom"]

    def __init__(
        self,
        atom_positions: Float[Array, "n_atoms 3"] | Float[np.ndarray, "n_atoms 3"],
        atom_strengths: (
            Float[Array, "n_atoms n_gaussians_per_atom"]
            | Float[np.ndarray, "n_atoms n_gaussians_per_atom"]
        ),
        atom_b_factors: (
            Float[Array, "n_atoms n_gaussians_per_atom"]
            | Float[np.ndarray, "n_atoms n_gaussians_per_atom"]
        ),
    ):
        """**Arguments:**

        - `atom_positions`: The coordinates of the atoms in units of angstroms.
        - `atom_strengths`: The strength for each atom and for each gaussian per atom.
                            This has units of angstroms.
        - `atom_b_factors`: The variance (up to numerical constants) for each atom and
                            for each gaussian per atom. This has units of angstroms
                            squared.
        """
        self.atom_positions = jnp.asarray(atom_positions)
        self.atom_strengths = jnp.asarray(atom_strengths)
        self.atom_b_factors = jnp.asarray(atom_b_factors)

    @override
    def as_real_voxel_grid(
        self,
        coordinate_grid_in_angstroms: Float[Array, "z_dim y_dim x_dim 3"],
        *,
        batch_size: Optional[int] = None,
        progress_bar: bool = False,
        print_every: Optional[int] = None,
    ) -> Float[Array, "z_dim y_dim x_dim"]:
        """Return a voxel grid in real space of the potential.

        **Arguments:**

        - `coordinate_grid_in_angstroms`: The coordinate system of the grid.
        - `batch_size`: The number of atoms over which to compute the potential
                        in parallel.

        **Returns:**

        The rescaled potential $U(\\mathbf{x})$ as a voxel grid evaluated on the
        `coordinate_grid_in_angstroms`.
        """  # noqa: E501
        return _build_real_space_voxels_from_atoms(
            self.atom_positions,
            self.atom_strengths,
            self.atom_b_factors,
            coordinate_grid_in_angstroms,
            batch_size=batch_size,
            progress_bar=progress_bar,
            print_every=print_every,
        )


class AbstractTabulatedAtomicPotential(AbstractAtomicPotential, strict=True):
    atom_b_factors: eqx.AbstractVar[Float[Array, " n_atoms"]]


class PengTabulatedAtomicPotential(AbstractTabulatedAtomicPotential, strict=True):
    """The scattering potential parameterized as a mixture of five gaussians
    per atom, through work by Lian-Mao Peng.

    Parameters `scattering_factor_a` and `scattering_factor_b` are referred
    to as $a_i$ and $b_i$ respectively in "Robust Parameterization of Elastic
    and Absorptive Electron Atomic Scattering Factors" by Peng et al. (1996).

    !!! info
        In order to load a `PengTabulatedAtomicPotential` from tabulated
        scattering factors, use the `cryojax.constants` submodule.

        ```python
        from cryojax.constants import (
            peng_element_scattering_factor_parameter_table,
            get_tabulated_scattering_factor_parameters,
        )
        from cryojax.data import read_atoms_from_pdb
        from cryojax.simulator import PengTabulatedAtomicPotential

        # Load positions of atoms and one-hot encoded atom names
        atom_positions, atom_identities = read_atoms_from_pdb(...)
        scattering_factor_a, scattering_factor_b = get_tabulated_scattering_factor_parameters(
            atom_identities, peng_element_scattering_factor_parameter_table
        )
        potential = PengTabulatedAtomicPotential(
            atom_positions, scattering_factor_a, scattering_factor_b
        )
        ```

    **References:**

    - Peng, L-M. "Electron atomic scattering factors and scattering potentials of crystals."
      Micron 30.6 (1999): 625-648.
    - Peng, L-M., et al. "Robust parameterization of elastic and absorptive electron atomic
      scattering factors." Acta Crystallographica Section A: Foundations of Crystallography
      52.2 (1996): 257-276.
    """  # noqa: E501

    atom_positions: Float[Array, "n_atoms 3"]
    scattering_factor_a: Float[Array, "n_atoms 5"]
    scattering_factor_b: Float[Array, "n_atoms 5"]
    atom_b_factors: Float[Array, " n_atoms"]

    def __init__(
        self,
        atom_positions: Float[Array, "n_atoms 3"] | Float[np.ndarray, "n_atoms 3"],
        scattering_factor_a: Float[Array, "n_atoms 5"] | Float[np.ndarray, "n_atoms 5"],
        scattering_factor_b: Float[Array, "n_atoms 5"] | Float[np.ndarray, "n_atoms 5"],
        atom_b_factors: Optional[
            Float[Array, " n_atoms"] | Float[np.ndarray, " n_atoms"]
        ] = None,
    ):
        """**Arguments:**

        - `atom_positions`: The coordinates of the atoms in units of angstroms.
        - `scattering_factor_a`: The scattering factors parameter "$a_i$" from
                                 Peng et al. (1996)
        - `scattering_factor_b`: The scattering factors parameter "$b_i$" from
                                 Peng et al. (1996)
        - `atom_b_factors`: The B factors applied to each atom.
        """
        self.atom_positions = jnp.asarray(atom_positions)
        self.scattering_factor_a = jnp.asarray(scattering_factor_a)
        self.scattering_factor_b = error_if_not_positive(jnp.asarray(scattering_factor_b))
        if atom_b_factors is None:
            n_atoms = atom_positions.shape[0]
            atom_b_factors = jnp.full((n_atoms,), 0.0, dtype=float)
        self.atom_b_factors = error_if_negative(jnp.asarray(atom_b_factors))

    @override
    def as_real_voxel_grid(
        self,
        coordinate_grid_in_angstroms: Float[Array, "z_dim y_dim x_dim 3"],
        *,
        batch_size: Optional[int] = None,
        progress_bar: bool = False,
        print_every: Optional[int] = None,
    ) -> Float[Array, "z_dim y_dim x_dim"]:
        """Return a voxel grid in real space of the potential.

        In the notation of Peng et al. (1996), tabulated `scattering_factor_a` and
        `scattering_factor_b` parameterize the elastic electron scattering factors,
        defined as

        $$f^{(e)}(\\mathbf{q}) = \\sum\\limits_{i = 1}^5 a_i \\exp(- b_i |\\mathbf{q}|^2),$$

        where $a_i$ is the `scattering_factor_a` and $b_i$ is the `scattering_factor_b`.
        Under usual scattering approximations (i.e. the first-born approximation),
        the rescaled electrostatic potential energy $U(\\mathbf{x})$ is then given by
        $32 \\pi \\mathcal{F}^{-1}[f^{(e)}](2 \\mathbf{x})$, which is computed analytically as

        $$U(\\mathbf{x}) = \\sum\\limits_{i = 1}^5 \\frac{4 \\pi a_i}{(2\\pi (b_i / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{x}|^2}{2 (b_i / 8 \\pi^2)}).$$

        We additionally give the option to modulate this expression with a constant B-factor,
        giving the expression

        $$U(\\mathbf{x}) = \\sum\\limits_{i = 1}^5 \\frac{4 \\pi a_i}{(2\\pi ((b_i + B / 4) / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{x}|^2}{2 ((b_i + B / 4) / 8 \\pi^2)}),$$

        where $B$ is the `atom_b_factors`.

        **Arguments:**

        - `coordinate_grid_in_angstroms`: The coordinate system of the grid.
        - `batch_size`: The number of atoms over which to compute the potential
                        in parallel with `jax.vmap`.

        **Returns:**

        The rescaled potential $U(\\mathbf{x})$ as a voxel grid evaluated on the
        `coordinate_grid_in_angstroms`.
        """  # noqa: E501
        return _build_real_space_voxels_from_atoms(
            self.atom_positions,
            self.scattering_factor_a,
            self.scattering_factor_b + self.atom_b_factors[:, None] / 4,
            coordinate_grid_in_angstroms,
            batch_size=batch_size,
            progress_bar=progress_bar,
            print_every=print_every,
        )


def _evaluate_3d_real_space_gaussian(
    coordinate_grid_in_angstroms: Float[Array, "z_dim y_dim x_dim 3"],
    atom_position: Float[Array, "3"],
    a: Float[Array, ""],
    b: Float[Array, ""],
) -> Float[Array, "z_dim y_dim x_dim"]:
    """Evaluate a gaussian on a 3D grid.

    **Arguments:**

    - `coordinate_grid`: The coordinate system of the grid.
    - `pos`: The center of the gaussian.
    - `a`: A scale factor.
    - `b`: The scale of the gaussian.

    **Returns:**

    The potential of the gaussian on the grid.
    """
    b_inverse = 4.0 * jnp.pi / b
    sq_distances = jnp.sum(
        b_inverse * (coordinate_grid_in_angstroms - atom_position) ** 2, axis=-1
    )
    return 4 * jnp.pi * jnp.exp(-jnp.pi * sq_distances) * a * b_inverse ** (3.0 / 2.0)


def _evaluate_3d_atom_potential(
    coordinate_grid_in_angstroms: Float[Array, "z_dim y_dim x_dim 3"],
    atom_position: Float[Array, "3"],
    atomic_as: Float[Array, " n_scattering_factors"],
    atomic_bs: Float[Array, " n_scattering_factors"],
) -> Float[Array, "z_dim y_dim x_dim"]:
    """Evaluates the electron potential of a single atom on a 3D grid.

    **Arguments:**

    - `coordinate_grid_in_angstroms`: The coordinate system of the grid.
    - `atom_position`: The location of the atom.
    - `atomic_as`: The intensity values for each gaussian in the atom.
    - `atomic_bs`: The inverse scale factors for each gaussian in the atom.

    **Returns:**

    The potential of the atom evaluated on the grid.
    """
    eval_fxn = jax.vmap(_evaluate_3d_real_space_gaussian, in_axes=(None, None, 0, 0))
    return jnp.sum(
        eval_fxn(coordinate_grid_in_angstroms, atom_position, atomic_as, atomic_bs),
        axis=0,
    )


@eqx.filter_jit
def _build_real_space_voxels_from_atoms(
    atom_positions: Float[Array, "n_atoms 3"],
    ff_a: Float[Array, "n_atoms n_gaussians_per_atom"],
    ff_b: Float[Array, "n_atoms n_gaussians_per_atom"],
    coordinate_grid_in_angstroms: Float[Array, "z_dim y_dim x_dim 3"],
    *,
    batch_size: Optional[int] = None,
    progress_bar: bool = False,
    print_every: Optional[int] = None,
) -> Float[Array, "z_dim y_dim x_dim"]:
    """
    Build a voxel representation of an atomic model.

    **Arguments**

    - `atom_coords`: The coordinates of the atoms.
    - `ff_a`: Intensity values for each Gaussian in the atom
    - `ff_b` : The inverse scale factors for each Gaussian in the atom
    - `coordinate_grid` : The coordinates of each voxel in the grid.

    **Returns:**

    The voxel representation of the atomic model.
    """
    voxel_grid_buffer = jnp.zeros(coordinate_grid_in_angstroms.shape[:-1])

    # TODO: Look into forcing JAX to do in-place updates
    # Below is a first attempt at this with `donate_argnums`, however
    # equinox.internal.while_loop / equinox.internal.scan could also be
    # options
    @partial(jax.jit, donate_argnums=1)
    def brute_force_body_fun(atom_index, potential):
        return potential + _evaluate_3d_atom_potential(
            coordinate_grid_in_angstroms,
            atom_positions[atom_index],
            ff_a[atom_index],
            ff_b[atom_index],
        )

    def evaluate_3d_atom_potential_batch(atom_index_batch):
        vmap_evaluate_3d_atom_potential = jax.vmap(
            _evaluate_3d_atom_potential, in_axes=[None, 0, 0, 0]
        )
        return jnp.sum(
            vmap_evaluate_3d_atom_potential(
                coordinate_grid_in_angstroms,
                jnp.take(atom_positions, atom_index_batch, axis=0),
                jnp.take(ff_a, atom_index_batch, axis=0),
                jnp.take(ff_b, atom_index_batch, axis=0),
            ),
            axis=0,
        )

    @partial(jax.jit, donate_argnums=1)
    def batched_body_fun(iteration_index, potential):
        atom_index_batch = jnp.linspace(
            iteration_index * batch_size,
            (iteration_index + 1) * batch_size - 1,
            batch_size,  # type: ignore
            dtype=int,
        )
        return potential + evaluate_3d_atom_potential_batch(atom_index_batch)

    # Get the number of iterations of the loop (the number of atoms)
    n_atoms = atom_positions.shape[0]
    # Set the logic for the loop based on the batch size
    if batch_size is None:
        n_iterations = n_atoms
        body_fun = brute_force_body_fun
        if progress_bar:
            body_fun = fori_loop_tqdm_decorator(n_iterations, print_every)(body_fun)
        voxel_grid = jax.lax.fori_loop(0, n_atoms, body_fun, voxel_grid_buffer)
    else:
        n_iterations = n_atoms // batch_size
        body_fun = batched_body_fun
        if progress_bar:
            body_fun = fori_loop_tqdm_decorator(n_iterations, print_every)(body_fun)
        voxel_grid = jax.lax.fori_loop(0, n_iterations, body_fun, voxel_grid_buffer)
        if n_atoms % batch_size > 0:
            voxel_grid += evaluate_3d_atom_potential_batch(
                jnp.arange(n_atoms - n_atoms % batch_size, n_atoms)
            )

    return voxel_grid
