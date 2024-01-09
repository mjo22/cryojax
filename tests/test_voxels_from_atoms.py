import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.simulator.density import VoxelGrid, VoxelCloud
from cryojax.simulator.density._voxel_density import (
    _build_real_space_voxels_from_atoms,
)
from cryojax.utils import ifftn, make_coordinates
from jax import config

config.update("jax_enable_x64", True)


def test_VoxelGrid_VoxelCloud_agreement(sample_pdb_path):
    """
    Integration test ensuring that the VoxelGrid and VoxelCloud classes
    produce comparable electron densities when loaded from PDB.
    """
    n_voxels_per_side = (128, 128, 128)
    voxel_size = 0.5

    # Load the PDB file into a VoxelGrid
    vg = VoxelGrid.from_pdb(
        sample_pdb_path,
        n_voxels_per_side=n_voxels_per_side,
        voxel_size=voxel_size,
    )
    # Since Voxelgrid is in Frequency space by default, we have to first
    # transform back into real space.
    vg_density = ifftn(vg.weights).real
    # The constructors each transpose in a unique way in order for
    # jax-finufft and the fourier slice theorem to match each other
    # and cisTEM. This operation undos the difference between both transposes
    vg_density = jnp.transpose(vg_density, axes=[1, 0, 2])
    # Ravel the grid
    vg_density = vg_density.ravel()

    vc = VoxelCloud.from_pdb(
        sample_pdb_path,
        n_voxels_per_side=n_voxels_per_side,
        voxel_size=voxel_size,
        mask_zeros=False,
    )

    np.testing.assert_allclose(vg_density, vc.weights, atol=1e-12)


class TestBuildRealSpaceVoxelsFromAtoms:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_write_positions(self, largest_atom):
        """
        Test that the maxima of the density are in the correct positions.
        """
        atom_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        num_atoms = atom_positions.shape[0]
        ff_a = np.array(
            num_atoms
            * [
                [1.0, 0.5],
            ]
        )
        ff_a[
            largest_atom
        ] += 0.5  # Give one atom more weight for testing purposes
        ff_a = jnp.array(ff_a)
        ff_b = jnp.array(
            num_atoms
            * [
                [0.3, 0.2],
            ]
        )

        # Build the coordinate system
        n_voxels_per_side = (128, 128, 128)
        voxel_size = 0.1
        coordinate_system = make_coordinates(n_voxels_per_side, voxel_size)

        # Build the density
        density = _build_real_space_voxels_from_atoms(
            atom_positions, ff_a, ff_b, coordinate_system
        )

        # Find the maximum
        maximum_index = jnp.argmax(density)
        maximum_position = coordinate_system.reshape(-1, 3)[maximum_index]

        # Check that the maximum is in the correct position
        assert jnp.allclose(maximum_position, atom_positions[largest_atom])

    def test_integral_is_correct(self):
        """
        Test that the maxima of the density are in the correct positions.
        """
        atom_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        num_atoms = atom_positions.shape[0]
        ff_a = np.array(
            num_atoms
            * [
                [1.0, 0.5],
            ]
        )

        ff_b = jnp.array(
            num_atoms
            * [
                [0.3, 0.2],
            ]
        )

        n_voxels_per_side = (128, 128, 128)
        voxel_size = 0.05
        coordinate_system = make_coordinates(n_voxels_per_side, voxel_size)

        # Build the density
        density = _build_real_space_voxels_from_atoms(
            atom_positions, ff_a, ff_b, coordinate_system
        )

        integral = jnp.sum(density) * voxel_size**3
        assert jnp.isclose(integral, jnp.sum(ff_a))
