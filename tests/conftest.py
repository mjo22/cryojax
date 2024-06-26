import os

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    import cryojax as cryojax
    import cryojax.simulator as cs
    from cryojax.image import operators as op, rfftn
    from cryojax.io import read_array_with_spacing_from_mrc


# jax.config.update("jax_numpy_dtype_promotion", "strict")
# jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sample_mrc_path():
    return os.path.join(os.path.dirname(__file__), "data", "3j9g_potential_ps4_4.mrc")


@pytest.fixture
def sample_subunit_mrc_path():
    return os.path.join(
        os.path.dirname(__file__), "data", "3j9g_subunit_potential_ps4_4.mrc"
    )


@pytest.fixture
def sample_pdb_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao.pdb")


@pytest.fixture
def toy_gaussian_cloud():
    atom_positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    num_atoms = atom_positions.shape[0]
    ff_a = jnp.array(
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
    return (atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size)


@pytest.fixture
def pixel_size():
    return 4.4


@pytest.fixture
def voltage_in_kilovolts():
    return 300.0


@pytest.fixture
def config(pixel_size, voltage_in_kilovolts):
    return cs.InstrumentConfig((65, 66), pixel_size, voltage_in_kilovolts, pad_scale=1.1)


@pytest.fixture
def projection_method():
    return cs.FourierSliceExtraction(interpolation_order=1)


@pytest.fixture
def potential(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(sample_mrc_path)
    return cs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=1.3
    )


@pytest.fixture
def filters(config):
    return op.LowpassFilter(config.padded_frequency_grid_in_pixels)


@pytest.fixture
def masks(config):
    return op.CircularCosineMask(
        config.padded_coordinate_grid_in_angstroms,
        radius_in_angstroms_or_pixels=20 * float(config.pixel_size),
        rolloff_width_in_angstroms_or_pixels=3 * float(config.pixel_size),
    )


@pytest.fixture
def transfer_theory():
    return cs.ContrastTransferTheory(ctf=cs.ContrastTransferFunction())


@pytest.fixture
def detector():
    return cs.PoissonDetector(cs.IdealDQE())


@pytest.fixture
def pose():
    return cs.EulerAnglePose(
        view_phi=30.0,
        view_theta=100.0,
        view_psi=-10.0,
        offset_x_in_angstroms=10.0,
        offset_y_in_angstroms=-5.0,
    )


@pytest.fixture
def specimen(potential, pose):
    return cs.SingleStructureEnsemble(potential, pose)


@pytest.fixture
def solvent():
    return cs.GaussianIce(op.Constant(0.001**2))


@pytest.fixture
def theory(specimen, projection_method, transfer_theory, solvent):
    return cs.WeakPhaseScatteringTheory(
        specimen, projection_method, transfer_theory, solvent
    )


@pytest.fixture
def theory_with_solvent(specimen, projection_method, transfer_theory, solvent):
    return cs.WeakPhaseScatteringTheory(
        specimen, projection_method, transfer_theory, solvent
    )


@pytest.fixture
def noiseless_model(config, theory):
    return cs.IntensityImagingPipeline(instrument_config=config, scattering_theory=theory)


@pytest.fixture
def noisy_model(config, theory_with_solvent, detector):
    return cs.ElectronCountingImagingPipeline(
        instrument_config=config,
        scattering_theory=theory_with_solvent,
        detector=detector,
    )


@pytest.fixture
def test_image(noisy_model):
    image = noisy_model.render(jr.PRNGKey(1234))
    return rfftn(image)
