import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

import cryojax.simulator as cxs
from cryojax.constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from cryojax.image import crop_to_shape, irfftn
from cryojax.io import read_atoms_from_pdb


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "pixel_size, shape",
    (
        (0.5, (64, 64)),
        (0.5, (63, 63)),
        (0.5, (64, 63)),
        (0.5, (64, 63)),
    ),
)
def test_projection_methods_no_pose(sample_pdb_path, pixel_size, shape):
    """Test that computing a projection in real
    space agrees with real-space, with no rotation. This mostly
    makes sure there are no numerical artifacts in fourier space
    interpolation and that volumes are read in real vs. fourier
    at the same orientation.
    """
    # Objects for imaging
    instrument_config = cxs.InstrumentConfig(
        shape,
        pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # Real vs fourier potentials
    dim = max(*shape)  # Make sure to use `padded_shape` here
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path, center=True, loads_b_factors=True
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )
    base_potential = cxs.PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
        b_factors=b_factors,
    )
    base_method = cxs.GaussianMixtureProjection(use_error_functions=True)

    real_voxel_grid = base_potential.as_real_voxel_grid((dim, dim, dim), pixel_size)
    other_potentials = [
        cxs.FourierVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
        make_spline_potential(real_voxel_grid, pixel_size),
        cxs.GaussianMixtureAtomicPotential(
            atom_positions,
            scattering_factor_parameters["a"],
            (scattering_factor_parameters["b"] + b_factors[:, None]) / (8 * jnp.pi**2),
        ),
    ]
    #     cxs.RealVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
    #     cxs.RealVoxelCloudPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
    # ]
    other_projection_methods = [
        cxs.FourierSliceExtraction(),
        cxs.FourierSliceExtraction(),
        base_method,
    ]
    #     cxs.NufftProjection(),
    #     cxs.NufftProjection(),
    # ]

    projection_by_gaussian_integration = compute_projection(
        base_potential, base_method, instrument_config
    )
    # fourier_projection_by_gaussian_integration = compute_fourier_projection(
    #     base_potential, base_method, instrument_config
    # )
    for potential, projection_method in zip(other_potentials, other_projection_methods):
        if isinstance(projection_method, cxs.NufftProjection):
            try:
                projection_by_other_method = compute_projection(
                    potential, projection_method, instrument_config
                )
                # fourier_projection_by_other_method = compute_fourier_projection(
                #     potential, projection_method, instrument_config
                # )
            except Exception as err:
                warnings.warn(
                    "Could not test projection method `NufftProjection` "
                    "This is most likely because `jax_finufft` is not installed. "
                    f"Error traceback is:\n{err}"
                )
                continue
        else:
            projection_by_other_method = compute_projection(
                potential, projection_method, instrument_config
            )
            # fourier_projection_by_other_method = compute_fourier_projection(
            #     potential, projection_method, instrument_config
            # )
        np.testing.assert_allclose(
            projection_by_gaussian_integration, projection_by_other_method, atol=1e-12
        )
        # np.testing.assert_allclose(
        #     fourier_projection_by_gaussian_integration,
        #     fourier_projection_by_other_method,
        #     atol=1e-12,
        # )


# @pytest.mark.parametrize(
#     "pixel_size, shape, padded_shape, grid_dim",
#     (
#         (0.25, (128, 128), (255, 256), 128),
#         (0.25, (128, 128), (253, 254), 128),
#         (0.25, (127, 127), (255, 254), 127),
#         (0.25, (127, 127), (257, 256), 127),
#     ),
# )
# def test_if_voxel_vs_atom_has_pixel_offset(
#     sample_pdb_path, pixel_size, shape, padded_shape, grid_dim
# ):
#     """Test that even after padding and cropping, there remains
#     no pixel-wise offset between atom and voxel representations
#     """
#     # Objects for imaging
#     instrument_config = cxs.InstrumentConfig(
#         shape, pixel_size, voltage_in_kilovolts=300.0, padded_shape=padded_shape
#     )
#     # Atom vs voxel potentials
#     atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
#         sample_pdb_path, center=True, loads_b_factors=True
#     )
#     scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
#         atom_identities, read_peng_element_scattering_factor_parameter_table()
#     )
#     atom_potential = cxs.PengAtomicPotential(
#         atom_positions,
#         scattering_factor_a=scattering_factor_parameters["a"],
#         scattering_factor_b=scattering_factor_parameters["b"],
#         b_factors=b_factors,
#     )
#     atom_method = cxs.GaussianMixtureProjection(use_error_functions=True)
#     dim = grid_dim
#     real_voxel_grid = atom_potential.as_real_voxel_grid((dim, dim, dim), pixel_size)
#     voxel_potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
#         real_voxel_grid, pixel_size
#     )
#     voxel_method = cxs.FourierSliceExtraction()

#     projection_by_atoms = compute_projection(
#         atom_potential, atom_method, instrument_config
#     )
#     projection_by_voxels = compute_projection(
#         voxel_potential, voxel_method, instrument_config
#     )

#     # from matplotlib import pyplot as plt

#     # fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
#     # im1 = axes[0].imshow(projection_by_atoms, aspect="auto")
#     # im2 = axes[1].imshow(projection_by_voxels, aspect="auto")
#     # im3 = axes[2].imshow(
#     #     np.abs(projection_by_atoms - projection_by_voxels),
#     #     aspect="auto",
#     # )
#     # fig.colorbar(im1, ax=axes[0])
#     # fig.colorbar(im2, ax=axes[1])
#     # fig.colorbar(im3, ax=axes[2])
#     # plt.show()
#     np.testing.assert_allclose(projection_by_atoms, projection_by_voxels, atol=1e-12)


# @pytest.mark.parametrize(
#     "shape, euler_pose_params, pad_scale",
#     (
#         ((128, 128), (2.5, -5.0, 0.0, 0.0, 0.0), 1),
#         ((127, 127), (2.5, -5.0, 0.0, 0.0, 0.0), 1),
#         ((128, 128), (0.0, 0.0, 10.0, -30.0, 60.0), 1),
#         ((127, 127), (0.0, 0.0, 10.0, -30.0, 60.0), 1),
#         ((128, 128), (2.5, -5.0, 10.0, -30.0, 60.0), 1),
#         ((127, 127), (2.5, -5.0, 10.0, -30.0, 60.0), 1),
#     ),
# )
# def test_projection_methods_with_pose(
#     sample_pdb_path, shape, euler_pose_params, pad_scale
# ):
#     """Test that computing a projection across different
#     methods agrees. This tests pose convention and accuracy
#     for real vs fourier, atoms vs voxels, etc.
#     """
#     # Objects for imaging
#     pixel_size = 0.25
#     instrument_config = cxs.InstrumentConfig(
#         shape, pixel_size, voltage_in_kilovolts=300.0, pad_scale=pad_scale
#     )
#     euler_pose = cxs.EulerAnglePose(*euler_pose_params)
#     # Real vs fourier potentials
#     dim = max(*shape)
#     atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
#         sample_pdb_path, center=True, loads_b_factors=True
#     )
# scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
#     atom_identities, read_peng_element_scattering_factor_parameter_table()
# )
# atom_potential = cxs.PengAtomicPotential(
#     atom_positions,
#     scattering_factor_a=scattering_factor_parameters["a"],
#     scattering_factor_b=scattering_factor_parameters["b"],
#     b_factors=b_factors,
# )
#     base_method = cxs.GaussianMixtureProjection(use_error_functions=True)

#     real_voxel_grid = base_potential.as_real_voxel_grid((dim, dim, dim), pixel_size)
#     other_potentials = [
#         cxs.FourierVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
#         cxs.FourierVoxelGridPotentialInterpolator.from_real_voxel_grid(
#             real_voxel_grid, pixel_size
#         ),
#         cxs.GaussianMixtureAtomicPotential(
#             atom_positions,
#             scattering_factor_parameters["a"],
#             (scattering_factor_parameters["b"] + b_factors[:, None]) / (8 * jnp.pi**2),
#         ),
#     ]
#     #     cxs.RealVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
#     #     cxs.RealVoxelCloudPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
#     # ]
#     other_projection_methods = [
#         cxs.FourierSliceExtraction(),
#         cxs.FourierSliceExtraction(),
#         base_method,
#     ]
#     #     cxs.NufftProjection(),
#     #     cxs.NufftProjection(),
#     # ]
#     tol = [(1e-7, 0), (1e-7, 0), (1e-7, 0)]

#     projection_by_gaussian_integration = compute_projection_at_pose(
#         base_potential, base_method, euler_pose, instrument_config
#     )
#     for idx, (potential, projection_method) in enumerate(
#         zip(other_potentials, other_projection_methods)
#     ):
#         if isinstance(projection_method, cxs.NufftProjection):
#             try:
#                 projection_by_other_method = compute_projection_at_pose(
#                     potential, projection_method, euler_pose, instrument_config
#                 )
#             except Exception as err:
#                 warnings.warn(
#                     "Could not test projection method `NufftProjection` "
#                     "This is most likely because `jax_finufft` is not installed. "
#                     f"Error traceback is:\n{err}"
#                 )
#                 continue
#         else:
#             projection_by_other_method = compute_projection_at_pose(
#                 potential, projection_method, euler_pose, instrument_config
#             )
#         from matplotlib import pyplot as plt

#         fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
#         im1 = axes[0].imshow(projection_by_gaussian_integration, aspect="auto")
#         im2 = axes[1].imshow(projection_by_other_method, aspect="auto")
#         im3 = axes[2].imshow(
#             np.abs(projection_by_gaussian_integration - projection_by_other_method),
#             aspect="auto",
#         )
#         axes[0].set(title="Real-space projection")
#         axes[1].set(title="Fourier-slice extraction")
#         axes[2].set(title="Residuals")
#         fig.colorbar(im1, ax=axes[0])
#         fig.colorbar(im2, ax=axes[1])
#         fig.colorbar(im3, ax=axes[2])
#         plt.show()
#         np.testing.assert_allclose(
#             projection_by_gaussian_integration,
#             projection_by_other_method,
#             atol=tol[idx][0],
#             rtol=tol[idx][1],
#         )


# def compute_fourier_projection_no_crop(
#     potential: cxs.AbstractPotentialRepresentation,
#     method: cxs.AbstractPotentialIntegrator,
#     config: cxs.InstrumentConfig,
# ) -> Array:
#     return method.compute_integrated_potential(
#         potential, config, outputs_real_space=False
#     )


@eqx.filter_jit
def compute_projection(
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


@eqx.filter_jit
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


@eqx.filter_jit
def make_spline_potential(real_voxel_grid, voxel_size):
    return cxs.FourierVoxelSplinePotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size
    )
