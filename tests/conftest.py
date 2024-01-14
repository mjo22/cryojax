import os
import pytest

import equinox as eqx
import jax.random as jr
from jax import config

import cryojax.simulator as cs
from cryojax.image import operators as op
from cryojax.inference import distributions as dist
from cryojax.image import rfftn

config.update("jax_enable_x64", True)


@pytest.fixture
def sample_mrc_path():
    return os.path.join(
        os.path.dirname(__file__), "data", "3j9g_bfm1_ps4_4.mrc"
    )


@pytest.fixture
def sample_subunit_mrc_path():
    return os.path.join(
        os.path.dirname(__file__), "data", "3j9g_subunit_bfm1_ps4_4.mrc"
    )


@pytest.fixture
def sample_pdb_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao.pdb")


@pytest.fixture
def manager():
    return cs.ImageManager(shape=(65, 66), pad_scale=1.1)


@pytest.fixture
def pixel_size():
    return 5.32


@pytest.fixture
def scattering(manager, pixel_size):
    return cs.FourierSliceExtract(manager, pixel_size=pixel_size)


@pytest.fixture
def density(sample_mrc_path):
    return cs.FourierVoxelGrid.from_file(sample_mrc_path, pad_scale=1.3)


@pytest.fixture
def stacked_density(density):
    return density.from_list([density for _ in range(3)])


@pytest.fixture
def filters(manager):
    return op.LowpassFilter(manager.padded_frequency_grid.get())
    # return None


@pytest.fixture
def masks(manager):
    return op.CircularMask(manager.coordinate_grid.get())


@pytest.fixture
def instrument():
    return cs.Instrument(
        optics=cs.CTFOptics(),
        exposure=cs.Exposure(
            scaling=op.Constant(1000.0), offset=op.ZeroMode(0.0)
        ),
        detector=cs.GaussianDetector(op.Constant(1.0)),
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
def specimen(density, pose):
    return cs.Specimen(density=density, pose=pose)


@pytest.fixture
def solvent():
    return cs.GaussianIce()


@pytest.fixture
def noiseless_model(scattering, specimen, instrument):
    instrument = eqx.tree_at(
        lambda ins: ins.detector, instrument, cs.NullDetector()
    )
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
    image = noisy_model.sample(jr.PRNGKey(1234), view=False)
    return rfftn(image)
