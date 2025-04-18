import jax
import numpy as np
import pytest
from jaxtyping import Array

import cryojax.simulator as cxs
from cryojax.image import crop_to_shape, irfftn
from cryojax.io import read_atoms_from_pdb


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "shape",
    ((64, 64), (63, 63), (63, 64), (64, 63)),
)
def test_projection_methods_no_rotation(sample_pdb_path, shape):
    """Test that computing a projection in real
    space agrees with real-space, with no rotation. This mostly
    makes sure there are no numerical artifacts in fourier space
    interpolation and that volumes are read in real vs. fourier
    at the same orientation.
    """
    # Objects for imaging
    pixel_size = 0.5
    instrument_config = cxs.InstrumentConfig(shape, pixel_size, 300.0)
    # Real vs fourier potentials
    dim = max(*shape)
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path, center=True, loads_b_factors=True
    )
    atom_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
    fourier_voxel_potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        atom_potential.as_real_voxel_grid((dim, dim, dim), pixel_size), pixel_size
    )
    fourier_slice_method = cxs.FourierSliceExtraction()
    gaussian_integration_method = cxs.GaussianMixtureProjection(use_error_functions=True)

    fourier_projection_by_fourier_slice = np.asarray(
        compute_fourier_projection(
            fourier_voxel_potential, fourier_slice_method, instrument_config
        )
    )
    fourier_projection_by_gaussian_integration = np.asarray(
        compute_fourier_projection(
            atom_potential, gaussian_integration_method, instrument_config
        )
    )
    projection_by_fourier_slice = np.asarray(
        compute_projection(
            fourier_voxel_potential, fourier_slice_method, instrument_config
        )
    )
    projection_by_gaussian_integration = np.asarray(
        compute_projection(atom_potential, gaussian_integration_method, instrument_config)
    )
    np.testing.assert_allclose(
        projection_by_gaussian_integration, projection_by_fourier_slice, atol=1e-12
    )
    np.testing.assert_allclose(
        fourier_projection_by_gaussian_integration,
        fourier_projection_by_fourier_slice,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "shape, euler_pose_params, pad_scale",
    (
        ((128, 128), (2.5, -5.0, 0.0, 0.0, 0.0), 1),
        ((127, 127), (2.5, -5.0, 0.0, 0.0, 0.0), 1),
        ((128, 128), (0.0, 0.0, 10.0, -30.0, 60.0), 1),
        ((127, 127), (0.0, 0.0, 10.0, -30.0, 60.0), 1),
        ((128, 128), (2.5, -5.0, 10.0, -30.0, 60.0), 1),
        ((127, 127), (2.5, -5.0, 10.0, -30.0, 60.0), 1),
    ),
)
def test_real_vs_fourier_with_rotation_and_translation(
    sample_pdb_path, shape, euler_pose_params, pad_scale
):
    """Test that computing a projection in real
    space agrees with real-space, with rotation and translation.
    This test makes sure pose conventions are the same and
    that there are no fourier artifacts after applying translational
    phase shifts.
    """
    # Objects for imaging
    pixel_size = 0.25
    instrument_config = cxs.InstrumentConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, pad_scale=pad_scale
    )
    euler_pose = cxs.EulerAnglePose(*euler_pose_params)
    # Real vs fourier potentials
    dim = max(*shape)
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path, center=True, loads_b_factors=True
    )
    atom_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
    fourier_voxel_potential = (
        cxs.FourierVoxelGridPotentialInterpolator.from_real_voxel_grid(
            atom_potential.as_real_voxel_grid((dim, dim, dim), pixel_size), pixel_size
        )
    )
    fourier_space_method = cxs.FourierSliceExtraction()
    real_space_method = cxs.GaussianMixtureProjection(use_error_functions=True)

    projection_by_fourier_method = np.asarray(
        compute_projection_at_pose(
            fourier_voxel_potential, fourier_space_method, euler_pose, instrument_config
        )
    )
    projection_by_real_method = np.asarray(
        compute_projection_at_pose(
            atom_potential, real_space_method, euler_pose, instrument_config
        )
    )
    # from matplotlib import pyplot as plt

    # fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
    # im1 = axes[0].imshow(projection_by_real_method, aspect="auto")
    # im2 = axes[1].imshow(projection_by_fourier_method, aspect="auto")
    # im3 = axes[2].imshow(
    #     np.abs(projection_by_real_method - projection_by_fourier_method), aspect="auto"
    # )
    # axes[0].set(title="Real-space projection")
    # axes[1].set(title="Fourier-slice extraction")
    # axes[2].set(title="Residuals")
    # fig.colorbar(im1, ax=axes[0])
    # fig.colorbar(im2, ax=axes[1])
    # fig.colorbar(im3, ax=axes[2])
    # plt.show()
    np.testing.assert_allclose(
        projection_by_real_method, projection_by_fourier_method, atol=1e-12
    )


@pytest.mark.parametrize(
    "shape, pad_scale",
    (
        ((128, 128), 1),
        ((127, 127), 1),
        ((128, 127), 1),
        ((128, 127), 1),
    ),
)
def test_atoms_vs_voxels_padding(sample_pdb_path, shape, pad_scale):
    """Test that computing a projection in real
    space agrees with real-space, with rotation and translation.
    This test makes sure pose conventions are the same and
    that there are no fourier artifacts after applying translational
    phase shifts.
    """
    # Objects for imaging
    pixel_size = 0.25
    instrument_config = cxs.InstrumentConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, pad_scale=pad_scale
    )
    # Real vs fourier potentials
    dim = max(*shape)
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path, center=True, loads_b_factors=True
    )
    atom_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
    fourier_voxel_potential = (
        cxs.FourierVoxelGridPotentialInterpolator.from_real_voxel_grid(
            atom_potential.as_real_voxel_grid((dim, dim, dim), pixel_size), pixel_size
        )
    )
    voxel_method = cxs.FourierSliceExtraction()
    atom_method = cxs.GaussianMixtureProjection(use_error_functions=True)

    projection_by_voxel_method = np.asarray(
        compute_projection(fourier_voxel_potential, voxel_method, instrument_config)
    )
    projection_by_atom_method = np.asarray(
        compute_projection(atom_potential, atom_method, instrument_config)
    )
    np.testing.assert_allclose(
        projection_by_atom_method, projection_by_voxel_method, atol=1e-12
    )


def compute_fourier_projection(
    potential: cxs.AbstractPotentialRepresentation,
    method: cxs.AbstractPotentialIntegrator,
    config: cxs.InstrumentConfig,
) -> Array:
    fourier_projection = method.compute_integrated_potential(
        potential, config, outputs_real_space=False
    )
    return crop_to_shape(
        irfftn(
            fourier_projection,
            s=config.padded_shape,
        ),
        config.shape,
    )


def compute_projection(
    potential: cxs.AbstractPotentialRepresentation,
    method: cxs.AbstractPotentialIntegrator,
    config: cxs.InstrumentConfig,
) -> Array:
    fourier_projection = method.compute_integrated_potential(
        potential, config, outputs_real_space=True
    )
    return crop_to_shape(
        irfftn(
            fourier_projection,
            s=config.padded_shape,
        ),
        config.shape,
    )


def compute_projection_at_pose(
    potential: cxs.AbstractPotentialRepresentation,
    method: cxs.AbstractPotentialIntegrator,
    pose: cxs.AbstractPose,
    config: cxs.InstrumentConfig,
) -> Array:
    rotated_potential = potential.rotate_to_pose(pose)
    fourier_projection = method.compute_integrated_potential(
        rotated_potential, config, outputs_real_space=False
    )
    translation_operator = pose.compute_translation_operator(
        config.padded_frequency_grid_in_angstroms
    )
    return crop_to_shape(
        irfftn(
            pose.translate_image(
                fourier_projection,
                translation_operator,
                config.padded_shape,
            ),
            s=config.padded_shape,
        ),
        config.shape,
    )
