import os
import pytest

from jax_2dtm.io import load_grid_as_cloud
from jax_2dtm.utils import ifft

from jax_2dtm.simulator import ScatteringConfig
from jax_2dtm.simulator import EulerPose, CTFOptics, WhiteNoise, ParameterState
from jax_2dtm.simulator import ScatteringImage, OpticsImage, GaussianImage


@pytest.fixture
def setup():
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_13pf_bfm1_ps5_28.mrc"
    )
    config = ScatteringConfig((81, 81), 5.32, eps=1e-4)
    cloud = load_grid_as_cloud(filename, config)
    return config, cloud


@pytest.fixture
def scattering_model(setup):
    config, cloud = setup
    state = ParameterState(pose=EulerPose())
    return ScatteringImage(config=config, cloud=cloud, state=state)


@pytest.fixture
def optics_model(setup):
    config, cloud = setup
    state = ParameterState(pose=EulerPose(), optics=CTFOptics())
    return OpticsImage(config=config, cloud=cloud, state=state)


@pytest.fixture
def noisy_model(setup):
    config, cloud = setup
    state = ParameterState(
        pose=EulerPose(), optics=CTFOptics(), noise=WhiteNoise()
    )
    return GaussianImage(config=config, cloud=cloud, state=state)


def test_shape(scattering_model):
    image = scattering_model()
    assert image.shape == scattering_model.config.shape


def test_normalization(optics_model):
    image = optics_model()
    assert pytest.approx(float(ifft(image).mean()), abs=1e-8) == 0.0
    assert pytest.approx(float(ifft(image).std()), abs=1e-6) == 1.0
