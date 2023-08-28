import numpy as np
from jax import config

import cryojax.simulator as cs
from cryojax.utils import fft

config.update("jax_enable_x64", True)


def test_deserialize_state_components(noisy_model):
    """Test deserialization on each component of the state."""
    state = noisy_model.state
    pose = cs.EulerPose.from_json(noisy_model.state.pose.to_json())
    ice = cs.ExponentialNoiseIce.from_json(noisy_model.state.ice.to_json())
    detector = cs.WhiteNoiseDetector.from_json(
        noisy_model.state.detector.to_json()
    )
    optics = cs.CTFOptics.from_json(noisy_model.state.optics.to_json())
    exposure = cs.UniformExposure.from_json(
        noisy_model.state.exposure.to_json()
    )
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
        state = cs.PipelineState.from_json(model.state.to_json())
        assert state.to_json() == model.state.to_json()
        np.testing.assert_allclose(
            model.replace(state=state)(), model(), rtol=1e-6
        )


def test_deserialize_specimen(setup):
    """Test specimen deserialization."""
    _, cloud = setup
    assert (
        cs.ElectronGrid.from_json(cloud.to_json()).to_json() == cloud.to_json()
    )
    # assert (
    #    ElectronCloud.from_json(cloud.to_json()).to_json() == cloud.to_json()
    # )


def test_deserialize_filters_and_masks(filters, masks):
    """Test model deserialization."""
    filters_and_masks = [*filters, *masks]
    types = [getattr(cs, f.__class__.__name__) for f in filters_and_masks]
    for idx, f in enumerate(filters_and_masks):
        test = types[idx].from_json(f.to_json())
        assert test.to_json() == f.to_json()
        np.testing.assert_allclose(test.compute(), f.compute(), rtol=1e-6)


def test_deserialize_model(scattering_model, optics_model, noisy_model):
    """Test model deserialization."""
    models = [scattering_model, optics_model, noisy_model]
    types = [getattr(cs, model.__class__.__name__) for model in models]
    for idx, model in enumerate(models):
        test_model = types[idx].from_json(model.to_json())
        assert model.to_json() == test_model.to_json()
        np.testing.assert_allclose(test_model(), model(), rtol=1e-6)
