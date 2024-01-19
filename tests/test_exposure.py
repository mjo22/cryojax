import pytest

from jax import config
import equinox as eqx
import numpy as np

from cryojax.image import operators as op
from cryojax.simulator import Exposure, NullExposure, NullIce

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["noisy_model", "noiseless_model"])
def test_scale(model, request):
    model = request.getfixturevalue(model)
    rescaled_model = eqx.tree_at(lambda x: x.solvent, model, NullIce())
    N, M = 10.0, 5.0
    exposure = Exposure(dose=op.Constant(N), radiation=op.Constant(M))
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

    np.testing.assert_allclose(rescaled_image, N * M * null_image)
