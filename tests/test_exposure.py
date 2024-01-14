import pytest

from jax import config
import equinox as eqx
import numpy as np

from cryojax.image import operators as op
from cryojax.simulator import Exposure, NullExposure

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["noisy_model", "noiseless_model"])
def test_rescale(model, request):
    model = request.getfixturevalue(model)
    N, mu = 10.0, 0.5
    exposure = Exposure(scaling=op.Constant(N), offset=op.ZeroMode(mu))
    null_exposure = NullExposure()
    # Create null model
    rescaled_model = eqx.tree_at(
        lambda x: x.instrument.exposure, model, exposure
    )
    null_model = eqx.tree_at(
        lambda x: x.instrument.exposure, model, null_exposure
    )
    # Compute images
    null_image = null_model.render(view_cropped=False)
    rescaled_image = rescaled_model.render(view_cropped=False)

    np.testing.assert_allclose(rescaled_image, N * null_image + mu)
