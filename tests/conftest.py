import os
import pytest

import equinox as eqx
import jax.random as jr
from jax import config

import cryojax.simulator as cs
from cryojax.utils import fftn

config.update("jax_enable_x64", True)


@pytest.fixture
def manager():
    return cs.ImageManager(shape=(81, 82))


@pytest.fixture
def pixel_size():
    return 5.32


@pytest.fixture
def scattering(manager, pixel_size):
    return cs.FourierSliceExtract(manager, pixel_size=pixel_size)


@pytest.fixture
def density():
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_monomer_bfm1_ps5_28.mrc"
    )
    return cs.VoxelGrid.from_file(filename)


@pytest.fixture
def sample_pdb_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao.pdb")


@pytest.fixture
def filters(manager):
    return cs.LowpassFilter(manager)
    # return None


@pytest.fixture
def masks(manager):
    return cs.CircularMask(manager)


@pytest.fixture
def instrument(pixel_size):
    return cs.Instrument(
        optics=cs.CTFOptics(),
        exposure=cs.UniformExposure(N=1000, mu=0.0),
        detector=cs.GaussianDetector(cs.Constant(1.0)),
    )


@pytest.fixture
def ensemble(density):
    return cs.Ensemble(
        density=density,
        pose=cs.EulerPose(degrees=False),
    )


@pytest.fixture
def solvent():
    return cs.GaussianIce()


@pytest.fixture
def noiseless_model(scattering, ensemble, instrument):
    instrument = eqx.tree_at(
        lambda ins: ins.detector, instrument, cs.NullDetector()
    )
    return cs.ImagePipeline(
        scattering=scattering, ensemble=ensemble, instrument=instrument
    )


@pytest.fixture
def noisy_model(scattering, ensemble, instrument, solvent):
    return cs.ImagePipeline(
        scattering=scattering,
        ensemble=ensemble,
        instrument=instrument,
        solvent=solvent,
    )


@pytest.fixture
def filtered_model(scattering, ensemble, instrument, solvent, filters):
    return cs.ImagePipeline(
        scattering=scattering,
        ensemble=ensemble,
        instrument=instrument,
        solvent=solvent,
        filter=filters,
    )


@pytest.fixture
def filtered_and_masked_model(
    scattering, ensemble, instrument, solvent, filters, masks
):
    return cs.ImagePipeline(
        scattering=scattering,
        ensemble=ensemble,
        instrument=instrument,
        solvent=solvent,
        filter=filters,
        mask=masks,
    )


@pytest.fixture
def test_image(noisy_model):
    image = noisy_model.sample(jr.PRNGKey(1234), view=False)
    return fftn(image)


@pytest.fixture
def likelihood_model(noisy_model):
    return cs.IndependentFourierGaussian(noisy_model)


@pytest.fixture
def likelihood_model_with_custom_variance(noiseless_model):
    return cs.IndependentFourierGaussian(
        noiseless_model, noise=cs.GaussianNoise(variance=cs.Constant(1.0))
    )
