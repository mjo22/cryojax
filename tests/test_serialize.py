from .test_pipeline import setup, scattering_model, noisy_model, optics_model

import numpy as np

from cryojax.simulator import (
    EulerPose,
    WhiteNoise,
    CTFOptics,
    Intensity,
    ParameterState,
)


def test_deserialize_state_components(noisy_model):
    """Test deserialization on each component of the state."""
    state = noisy_model.state
    pose = EulerPose.from_json(noisy_model.state.pose.to_json())
    noise = WhiteNoise.from_json(noisy_model.state.noise.to_json())
    optics = CTFOptics.from_json(noisy_model.state.optics.to_json())
    intensity = Intensity.from_json(noisy_model.state.intensity.to_json())
    model = noisy_model.replace(
        state=state.replace(
            pose=pose, noise=noise, optics=optics, intensity=intensity
        )
    )
    np.testing.assert_allclose(noisy_model(), model())


def test_deserialize_state(scattering_model, optics_model, noisy_model):
    """Test ParameterState deserialization"""
    models = [scattering_model, optics_model, noisy_model]
    for model in models:
        state = ParameterState.from_json(model.state.to_json())
        np.testing.assert_allclose(model.update(state)(), model())
