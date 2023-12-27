import os
import pytest

import jax.random as jr
from jax import config

import cryojax.simulator as cs
from cryojax.io import load_fourier_grid

config.update("jax_enable_x64", True)


@pytest.fixture()
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
def weights_and_coordinates():
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_monomer_bfm1_ps5_28.mrc"
    )
    return load_fourier_grid(filename)


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
        exposure=cs.UniformExposure(N=1e5, mu=1.0),
        detector=cs.GaussianDetector(pixel_size=pixel_size),
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
def noisy_model(scattering, ensemble, instrument, solvent, filters, masks):
    return cs.ImagePipeline(
        scattering=scattering,
        ensemble=ensemble,
        instrument=instrument,
        solvent=solvent,
        filter=filters,
        mask=masks,
    )


@pytest.fixture
def maskless_model(scattering, ensemble, instrument, solvent, filters):
    return cs.ImagePipeline(
        scattering=scattering,
        ensemble=ensemble,
        instrument=instrument,
        solvent=solvent,
        filter=filters,
    )


@pytest.fixture
def test_image(noisy_model):
    return noisy_model.sample(jr.split(jr.PRNGKey(1234), num=2))


@pytest.fixture
def likelihood_model(
    scattering, ensemble, instrument, solvent, filters, masks
):
    return cs.GaussianImage(
        scattering=scattering,
        ensemble=ensemble,
        instrument=instrument,
        solvent=solvent,
        filter=filters,
        mask=masks,
    )
