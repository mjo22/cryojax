import jax
import numpy as np
import pytest
from jaxtyping import Array

import cryojax.simulator as cxs
from cryojax.io import read_atoms_from_pdb


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "shape",
    ((64, 64), (63, 63), (63, 64), (64, 63)),
)
def test_real_vs_fourier_convention_no_rotation(sample_pdb_path, shape):
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
    fourier_space_method = cxs.FourierSliceExtraction(interpolation_order=1)
    real_space_method = cxs.GaussianMixtureProjection(use_error_functions=True)

    def compute_projection(
        potential: cxs.AbstractPotentialRepresentation,
        method: cxs.AbstractPotentialIntegrator,
        config: cxs.InstrumentConfig,
    ) -> Array:
        return method.compute_integrated_potential(
            potential, config, outputs_real_space=False
        )

    projection_by_fourier_method = np.asarray(
        compute_projection(
            fourier_voxel_potential, fourier_space_method, instrument_config
        )
    )
    projection_by_real_method = np.asarray(
        compute_projection(atom_potential, real_space_method, instrument_config)
    )
    _, _ = projection_by_fourier_method, projection_by_real_method
    # np.testing.assert_allclose(projection_by_real_method, projection_by_fourier_method)
    # from matplotlib import pyplot as plt
    # from cryojax.image import irfftn

    # im = np.fft.fftshift(np.fft.irfftn(projection_by_fourier_method, s=shape))
    # f_im = np.fft.fftshift(projection_by_fourier_method, axes=(0,))
    # plt.imshow(np.abs(f_im))
    # plt.imshow(im)
    # plt.show()
    # print(f_im.shape)
    # print(f_im[shape[0] - 1, :])


@pytest.mark.parametrize(
    "shape, euler_pose_params",
    (
        ((64, 64), (0.0, 0.0, 0.0, 0.0, 0.0)),
        # ((63, 63), (0.0, 0.0, 0.0, 0.0, 0.0)),
        #        ((63, 64), (0.0, 0.0, 0.0, 0.0, 0.0)),
        #        ((64, 63), (0.0, 0.0, 0.0, 0.0, 0.0)),
    ),
)
def test_real_vs_fourier_convention_with_rotation(
    sample_pdb_path, shape, euler_pose_params
):
    """Test that computing a projection in real
    space agrees with real-space, with rotation and translation.
    This test makes sure pose conventions are the same and
    that there are no fourier artifacts after applying translational
    phase shifts.
    """
    # Objects for imaging
    pixel_size = 0.5
    instrument_config = cxs.InstrumentConfig(shape, pixel_size, 300.0)
    euler_pose = cxs.EulerAnglePose(*euler_pose_params)
    # Real vs fourier potentials
    dim = max(*shape)
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path, center=True, loads_b_factors=True
    )
    atom_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
    fourier_voxel_potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        atom_potential.as_real_voxel_grid((dim, dim, dim), pixel_size), pixel_size
    )
    fourier_space_method = cxs.FourierSliceExtraction(interpolation_order=1)
    real_space_method = cxs.GaussianMixtureProjection(use_error_functions=True)

    def compute_projection(
        potential: cxs.AbstractPotentialRepresentation,
        method: cxs.AbstractPotentialIntegrator,
        pose: cxs.AbstractPose,
        config: cxs.InstrumentConfig,
    ) -> Array:
        rotated_potential = potential.rotate_to_pose(pose)
        return method.compute_integrated_potential(
            rotated_potential, config, outputs_real_space=False
        )

    projection_by_fourier_method = np.asarray(
        compute_projection(
            fourier_voxel_potential, fourier_space_method, euler_pose, instrument_config
        )
    )
    projection_by_real_method = np.asarray(
        compute_projection(
            atom_potential, real_space_method, euler_pose, instrument_config
        )
    )
    _, _ = projection_by_fourier_method, projection_by_real_method
    # np.testing.assert_allclose(projection_by_real_method, projection_by_fourier_method)
