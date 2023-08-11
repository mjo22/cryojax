import os
import pytest

import numpy as np
from jax import random
from jax import config

from cryojax.simulator import NufftScattering, ElectronCloud
from cryojax.simulator import (
    EulerPose,
    ExponentialNoiseIce,
    CTFOptics,
    UniformExposure,
    WhiteNoiseDetector,
    PipelineState,
)
from cryojax.simulator import ScatteringImage, OpticsImage, GaussianImage

config.update("jax_enable_x64", True)


@pytest.fixture
def setup():
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_13pf_bfm1_ps5_28.mrc"
    )
    scattering = NufftScattering((81, 81), 5.32, eps=1e-4)
    cloud = ElectronCloud.from_file(filename)
    return scattering, cloud


@pytest.fixture
def scattering_model(setup):
    scattering, cloud = setup
    state = PipelineState(pose=EulerPose(), exposure=UniformExposure())
    return ScatteringImage(scattering=scattering, specimen=cloud, state=state)


@pytest.fixture
def optics_model(setup):
    scattering, cloud = setup
    state = PipelineState(
        pose=EulerPose(), optics=CTFOptics(), exposure=UniformExposure()
    )
    return OpticsImage(scattering=scattering, specimen=cloud, state=state)


@pytest.fixture
def noisy_model(setup):
    scattering, cloud = setup
    state = PipelineState(
        pose=EulerPose(),
        ice=ExponentialNoiseIce(key=random.PRNGKey(seed=1)),
        optics=CTFOptics(),
        exposure=UniformExposure(),
        detector=WhiteNoiseDetector(key=random.PRNGKey(seed=0)),
    )
    return GaussianImage(scattering=scattering, specimen=cloud, state=state)


def test_shape(scattering_model):
    image = scattering_model()
    assert image.shape == scattering_model.scattering.shape


def test_update(noisy_model):
    offset_x, view_phi, voltage, N, alpha, xi, pixel_size = (
        50.23,
        np.pi / 2.2,
        250.23,
        0.77,
        0.676,
        12342.0,
        100.0,
    )
    params = dict(
        offset_x=offset_x,
        view_phi=view_phi,
        voltage=voltage,
        N=N,
        alpha=alpha,
        pixel_size=pixel_size,
        xi=xi,
    )
    scattering = noisy_model.scattering.update(**params)
    state = noisy_model.state.update(**params)
    model = noisy_model.update(**params)
    # Test scatttering update
    assert pixel_size == scattering.pixel_size
    # Test state update
    assert offset_x == state.pose.offset_x
    assert view_phi == state.pose.view_phi
    assert voltage == state.optics.voltage
    assert N == state.exposure.N
    assert alpha == state.detector.alpha
    assert xi == state.ice.xi
    # Test model update
    assert pixel_size == model.scattering.pixel_size
    assert offset_x == model.state.pose.offset_x
    assert view_phi == model.state.pose.view_phi
    assert voltage == model.state.optics.voltage
    assert N == model.state.exposure.N
    assert alpha == model.state.detector.alpha
    assert xi == model.state.ice.xi
