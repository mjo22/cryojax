import pytest

from jax import config

from cryojax.utils import irfft
from cryojax.simulator import NullExposure

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "rescaled_model", ["scattering_model", "optics_model"]
)
def test_rescale(rescaled_model, request):
    rescaled_model = request.getfixturevalue(rescaled_model)
    exposure = rescaled_model.state.exposure
    mu, N = exposure.mu, exposure.N
    # Initialize null model
    null_model = rescaled_model.update(exposure=NullExposure())
    # Compute images
    null_image = irfft(null_model.render(view=False))
    rescaled_image = irfft(rescaled_model.render(view=False))
    assert (
        pytest.approx(rescaled_image.sum().item())
        == (N * null_image + mu).sum().item()
    )
