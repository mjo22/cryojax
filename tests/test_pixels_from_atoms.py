import jax.numpy as jnp
import numpy as np
import pytest
from jax import config
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.coordinates import make_coordinate_grid
    from cryojax.image import irfftn
    from cryojax.io import read_atoms_from_pdb
    from cryojax.simulator import (
        GaussianMixtureAtomicPotential,
        GaussianMixtureProjection,
        InstrumentConfig,
        PengAtomicPotential,
    )


config.update("jax_enable_x64", True)


def test_atom_potential_integrator(sample_pdb_path):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path, get_b_factors=True
    )
    atom_potential = PengAtomicPotential(atom_positions, atom_identities, b_factors)

    shape = (64, 64)
    pixel_size = 0.5

    potential_integrator = GaussianMixtureProjection(upsampling_factor=2)
    # # ... and the configuration of the imaging instrument
    instrument_config = InstrumentConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # ... compute the integrated volumetric_potential
    fourier_integrated_potential = (
        potential_integrator.compute_fourier_integrated_potential(
            atom_potential, instrument_config, batch_size=1
        )
    )
    projection = irfftn(fourier_integrated_potential)

    assert projection.shape == shape


def test_downsampled_gmm_potential_agreement(sample_pdb_path):
    """Integration test ensuring that rasterized voxel grids roughly
    agree with downsampled versions.
    """
    atom_positions, atom_identities = read_atoms_from_pdb(sample_pdb_path)
    atom_potential = PengAtomicPotential(atom_positions, atom_identities)

    # Parameters for rasterization
    shape = (128, 128)
    pixel_size = 0.25

    # Downsampling parameters
    downsampling_factor = 2
    downsampled_shape = (
        int(shape[0] / downsampling_factor),
        int(shape[1] / downsampling_factor),
    )
    downsampled_pixel_size = pixel_size * downsampling_factor

    integrator_int_hires = GaussianMixtureProjection(
        upsampling_factor=downsampling_factor
    )
    integrator_int_lowres = GaussianMixtureProjection(upsampling_factor=1)
    # ... and the configuration of the imaging instrument
    instrument_config = InstrumentConfig(
        shape=downsampled_shape,
        pixel_size=downsampled_pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # ... compute the integrated volumetric_potential
    image_from_hires = integrator_int_hires.compute_fourier_integrated_potential(
        atom_potential, instrument_config, batch_size=1
    )
    image_lowres = integrator_int_lowres.compute_fourier_integrated_potential(
        atom_potential, instrument_config, batch_size=1
    )

    assert image_from_hires.shape == image_lowres.shape


def test_peng_vs_gmm_agreement(sample_pdb_path):
    """Integration test ensuring that Peng Potential and GMM potential agree when
    gaussians are identical"""

    # Load atoms and build potentials
    atom_positions, atom_identities = read_atoms_from_pdb(sample_pdb_path)
    atom_potential = PengAtomicPotential(atom_positions, atom_identities)

    gaussian_widths = atom_potential.scattering_factor_b
    gaussian_amplitudes = atom_potential.scattering_factor_a

    gmm_potential = GaussianMixtureAtomicPotential(
        atom_positions, gaussian_amplitudes, gaussian_widths
    )

    # Create instrument configuration
    shape = (64, 64)
    pixel_size = 0.5
    instrument_config = InstrumentConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )

    # Compute projections
    integrator = GaussianMixtureProjection(upsampling_factor=1)
    projection_gmm = integrator.compute_fourier_integrated_potential(
        gmm_potential, instrument_config, batch_size=1
    )
    projection_peng = integrator.compute_fourier_integrated_potential(
        atom_potential, instrument_config, batch_size=1
    )

    np.testing.assert_allclose(projection_gmm, projection_peng)


@pytest.mark.parametrize("batch_size", (2, 3, 4))
def test_batched_vs_non_batched_loop_agreement(sample_pdb_path, batch_size):
    shape = (64, 64)
    pixel_size = 0.5

    # Load the PDB file
    atom_positions, atom_elements = read_atoms_from_pdb(sample_pdb_path)
    # Load atomistic potential
    atomic_potential = PengAtomicPotential(atom_positions, atom_elements)
    # Build the instrument configuration
    instrument_config = InstrumentConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # Build the potential integrators
    integrator = GaussianMixtureProjection()

    # Compute projections
    projection_batching = integrator.compute_fourier_integrated_potential(
        atomic_potential, instrument_config, batch_size=batch_size
    )
    projection = integrator.compute_fourier_integrated_potential(
        atomic_potential, instrument_config, batch_size=1
    )
    np.testing.assert_allclose(projection, projection_batching)


class TestBuildRealSpaceVoxelsFromAtoms:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(self, toy_gaussian_cloud, largest_atom):
        """
        Test that the maxima of the potential are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud

        n_pixels_per_side = n_voxels_per_side[:2]
        ff_a = ff_a.at[largest_atom].add(1.0)
        coordinate_grid = make_coordinate_grid(n_pixels_per_side, voxel_size)

        # Build the potential
        atomic_potential = GaussianMixtureAtomicPotential(atom_positions, ff_a, ff_b)
        instrument_config = InstrumentConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the potential integrators
        integrator = GaussianMixtureProjection()
        # Compute projections
        projection = integrator.compute_fourier_integrated_potential(
            atomic_potential, instrument_config
        )
        projection = irfftn(projection)

        # Find the maximum
        maximum_index = jnp.argmax(projection)
        maximum_position = coordinate_grid.reshape(-1, 2)[maximum_index]

        # Check that the maximum is in the correct position
        assert jnp.allclose(maximum_position, atom_positions[largest_atom][:2])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        """
        Test that the maxima of the potential are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud

        n_pixels_per_side = n_voxels_per_side[:2]
        # Build the potential
        atomic_potential = GaussianMixtureAtomicPotential(atom_positions, ff_a, ff_b)
        instrument_config = InstrumentConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the potential integrators
        integrator = GaussianMixtureProjection()
        # Compute projections
        projection = integrator.compute_fourier_integrated_potential(
            atomic_potential, instrument_config, batch_size=1
        )
        projection = irfftn(projection)

        integral = jnp.sum(projection) * voxel_size**2
        assert jnp.isclose(integral, jnp.sum(4 * jnp.pi * ff_a))
