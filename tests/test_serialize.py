from .test_pipeline import setup, scattering_model, noisy_model, optics_model

import numpy as np
from jax import config

from cryojax.simulator import (
    EulerPose,
    WhiteNoiseDetector,
    ExponentialNoiseIce,
    CTFOptics,
    UniformExposure,
    PipelineState,
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
        np.testing.assert_allclose(model.replace(state=state)(), model())
