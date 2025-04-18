import jax
import numpy as np
import pytest
from jaxtyping import Array

import cryojax.simulator as cxs
from cryojax.image import crop_to_shape, irfftn
from cryojax.io import read_atoms_from_pdb


jax.config.update("jax_enable_x64", True)


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
                s=instrument_config.padded_shape,
            ),
            instrument_config.shape,
        )

    projection_by_voxel_method = np.asarray(
        compute_projection(fourier_voxel_potential, voxel_method, instrument_config)
    )
    projection_by_atom_method = np.asarray(
        compute_projection(atom_potential, atom_method, instrument_config)
    )
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
    im1 = axes[0].imshow(projection_by_atom_method, aspect="auto")
    im2 = axes[1].imshow(projection_by_voxel_method, aspect="auto")
    im3 = axes[2].imshow(
        np.abs(projection_by_atom_method - projection_by_voxel_method), aspect="auto"
    )
    axes[0].set(title="Projection w/ atom-based method")
    axes[1].set(title="Projection w/ voxel-based method")
    axes[2].set(title="Residuals")
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    plt.show()
    # np.testing.assert_allclose(
    #     projection_by_atom_method, projection_by_voxel_method, atol=1e-12
    # )
