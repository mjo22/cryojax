import os
import pytest
import numpy as np
import equinox as eqx
import jax.random as jr
from jax import config

import cryojax.simulator as cs
from cryojax.io import read_array_with_spacing_from_mrc
from cryojax.image import operators as op
from cryojax.image import rfftn

config.update("jax_enable_x64", True)


@pytest.fixture
def sample_mrc_path():
    return os.path.join(os.path.dirname(__file__), "data", "3j9g_bfm1_ps4_4.mrc")


@pytest.fixture
def sample_subunit_mrc_path():
    return os.path.join(
        os.path.dirname(__file__), "data", "3j9g_subunit_bfm1_ps4_4.mrc"
    )


@pytest.fixture
def sample_pdb_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao.pdb")


@pytest.fixture
def toy_gaussian_cloud():
    atom_positions = np.array(
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

    ff_b = np.array(
        num_atoms
        * [
            [0.3, 0.2],
        ]
    )

    n_voxels_per_side = (128, 128, 128)
    voxel_size = 0.05
    return (atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size)


@pytest.fixture
def pixel_size():
    return 4.4


@pytest.fixture
def config(pixel_size):
    return cs.ImageConfig((65, 66), pixel_size, pad_scale=1.1)


@pytest.fixture
def scattering(config):
    return cs.FourierSliceExtract(config, interpolation_order=1)


@pytest.fixture
def potential(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(sample_mrc_path)
    return cs.FourierVoxelGrid.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=1.3
    )


@pytest.fixture
def filters(config):
    return op.LowpassFilter(config.padded_frequency_grid.get())


@pytest.fixture
def masks(config):
    return op.CircularMask(
        config.padded_coordinate_grid_in_angstroms.get(),
        radius=20 * config.pixel_size,
    )


@pytest.fixture
def instrument():
    return cs.Instrument(
        optics=cs.CTFOptics(),
        detector=cs.GaussianDetector(electrons_per_angstrom_squared=op.Constant(10.0)),
    )


@pytest.fixture
def pose():
    return cs.EulerPose(
        view_phi=30.0,
        view_theta=100.0,
        view_psi=-10.0,
        offset_x=10.0,
        offset_y=-5.0,
    )


@pytest.fixture
def specimen(potential, pose):
    return cs.Specimen(potential=potential, pose=pose)


@pytest.fixture
def solvent():
    return cs.GaussianIce()


@pytest.fixture
def noiseless_model(scattering, specimen, instrument):
    instrument = eqx.tree_at(lambda ins: ins.detector, instrument, cs.NullDetector())
    return cs.ImagePipeline(
        scattering=scattering, specimen=specimen, instrument=instrument
    )


@pytest.fixture
def noisy_model(scattering, specimen, instrument, solvent):
    return cs.ImagePipeline(
        scattering=scattering,
        specimen=specimen,
        instrument=instrument,
        solvent=solvent,
    )


@pytest.fixture
def filtered_model(scattering, specimen, instrument, solvent, filters):
    return cs.ImagePipeline(
        scattering=scattering,
        specimen=specimen,
        instrument=instrument,
        solvent=solvent,
        filter=filters,
    )


@pytest.fixture
def filtered_and_masked_model(
    scattering, specimen, instrument, solvent, filters, masks
):
    return cs.ImagePipeline(
        scattering=scattering,
        specimen=specimen,
        instrument=instrument,
        solvent=solvent,
        filter=filters,
        mask=masks,
    )


@pytest.fixture
def test_image(noisy_model):
    image = noisy_model.sample(jr.PRNGKey(1234), view_cropped=False)
    return rfftn(image)
