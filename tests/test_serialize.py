from .test_pipeline import setup, noisy_model

import numpy as np

from jax_2dtm.simulator import EulerPose, WhiteNoise, CTFOptics, Intensity


def test_state_deserialize(noisy_model):
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
