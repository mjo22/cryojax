import os
import pytest

import numpy as np
import jax.numpy as jnp
from jax import random

from jax_2dtm.io import load_grid_as_cloud
from jax_2dtm.utils import ifft

from jax_2dtm.simulator import ScatteringConfig
from jax_2dtm.simulator import (
    EulerPose,
    CTFOptics,
    Intensity,
    WhiteNoise,
    ParameterState,
)
from jax_2dtm.simulator import ScatteringImage, OpticsImage, GaussianImage


@pytest.fixture
def setup():
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_13pf_bfm1_ps5_28.mrc"
    )
    config = ScatteringConfig((60, 60), 5.32, eps=1e-4)
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
    key = random.PRNGKey(seed=0)
    state = ParameterState(
        pose=EulerPose(), optics=CTFOptics(), noise=WhiteNoise(key=key)
    )
    return GaussianImage(config=config, cloud=cloud, state=state)


def test_shape(scattering_model):
    image = scattering_model()
    assert image.shape == scattering_model.config.shape


def test_update(noisy_model):
    offset_x, view_phi, voltage, N, sigma = (
        50.23,
        np.pi / 2.2,
        250.23,
        0.77,
        0.676,
    )
    params = dict(
        offset_x=offset_x, view_phi=view_phi, voltage=voltage, N=N, sigma=sigma
    )
    state = noisy_model.state.update(params)
    model = noisy_model.update(params)
    # Test state update
    assert offset_x == state.pose.offset_x
    assert view_phi == state.pose.view_phi
    assert voltage == state.optics.voltage
    assert N == state.intensity.N
    assert sigma == state.noise.sigma
    # Test model update
    assert offset_x == model.state.pose.offset_x
    assert view_phi == model.state.pose.view_phi
    assert voltage == model.state.optics.voltage
    assert N == model.state.intensity.N
    assert sigma == model.state.noise.sigma


def test_normalization(optics_model):
    image = optics_model()
    assert pytest.approx(float(ifft(image).mean()), rel=1e-6) == 0.0
    assert pytest.approx(float(ifft(image).std()), abs=1e-1) == 1.0


def test_rescale(optics_model):
    N1, N2 = optics_model.config.shape
    N, mu = 1.5, 0.5
    params = dict(N=N, mu=mu)
    image = optics_model(params)
    assert pytest.approx(float(image[N1 // 2, N2 // 2].real)) == (N1 * N2) * mu
    assert (
        pytest.approx(
            float(
                jnp.sqrt(
                    jnp.sum((image * jnp.conjugate(image)).real)
                    - (N1 * N2) * mu
                )
            ),
            rel=1e-2,
        )
        == (N1 * N2) * N
    )
    assert pytest.approx(float(ifft(image).mean()), rel=1e-1) == mu
    assert pytest.approx(float(ifft(image).std()), rel=1e-1) == N
