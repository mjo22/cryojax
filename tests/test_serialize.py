from .test_pipeline import setup, scattering_model, noisy_model, optics_model

import numpy as np
from jax import config

from cryojax.simulator import (
    ElectronGrid,
    FourierSliceScattering,
    ElectronCloud,
    NufftScattering,
    EulerPose,
    WhiteNoiseDetector,
    ExponentialNoiseIce,
    CTFOptics,
    UniformExposure,
    PipelineState,
    ScatteringImage,
    OpticsImage,
    GaussianImage,
)
from cryojax.utils import fft

config.update("jax_enable_x64", True)


def test_deserialize_state_components(noisy_model):
    """Test deserialization on each component of the state."""
    state = noisy_model.state
    pose = EulerPose.from_json(noisy_model.state.pose.to_json())
    ice = ExponentialNoiseIce.from_json(noisy_model.state.ice.to_json())
    detector = WhiteNoiseDetector.from_json(
        noisy_model.state.detector.to_json()
    )
    optics = CTFOptics.from_json(noisy_model.state.optics.to_json())
    exposure = UniformExposure.from_json(noisy_model.state.exposure.to_json())
    model = noisy_model.replace(
        state=state.replace(
            pose=pose,
            detector=detector,
            ice=ice,
            optics=optics,
            exposure=exposure,
        )
    )
    np.testing.assert_allclose(fft(noisy_model()), fft(model()))


def test_deserialize_state(scattering_model, optics_model, noisy_model):
    """Test PipelineState deserialization"""
    models = [scattering_model, optics_model, noisy_model]
    for model in models:
        state = PipelineState.from_json(model.state.to_json())
        np.testing.assert_allclose(
            model.replace(state=state)(), model(), rtol=1e-6
        )


def test_deserialize_specimen(setup):
    """Test specimen deserialization."""
    _, cloud = setup
    assert ElectronGrid.from_json(cloud.to_json()).to_json() == cloud.to_json()
    # assert (
    #    ElectronCloud.from_json(cloud.to_json()).to_json() == cloud.to_json()
    # )


def test_deserialize_model(scattering_model, optics_model, noisy_model):
    """Test model deserialization."""
    types = [ScatteringImage, OpticsImage, GaussianImage]
    models = [scattering_model, optics_model, noisy_model]
    for idx, model in enumerate(models):
        test_model = types[idx].from_json(model.to_json())
        np.testing.assert_allclose(test_model(), model(), rtol=1e-6)
