import pytest

from dataclasses import replace
from jax import config

from cryojax.simulator import NullExposure

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("rescaled_model", ["noisy_model", "noiseless_model"])
def test_rescale(rescaled_model, request):
    rescaled_model = request.getfixturevalue(rescaled_model)
    exposure = rescaled_model.instrument.exposure
    mu, N = exposure.mu, exposure.N
    # Create null model
    instrument = replace(rescaled_model.instrument, exposure=NullExposure())
    null_model = replace(rescaled_model, instrument=instrument)
    # Compute images
    null_image = null_model.render(view=False)
    rescaled_image = rescaled_model.render(view=False)
    assert (
        pytest.approx(rescaled_image.sum().item())
        == (N * null_image + mu).sum().item()
    )
