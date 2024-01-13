import pytest

import jax
import jax.numpy as jnp
import numpy as np
from cryojax.image import irfftn
from jax import config

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "model",
    [
        "noisy_model",
        "noiseless_model",
        "filtered_model",
        "filtered_and_masked_model",
    ],
)
def test_compute_with_filters_and_masks(model, request):
    model = request.getfixturevalue(model)
    key = jax.random.PRNGKey(1234)
    im1 = model.render(get_real=True, normalize=True)
    im2 = model.render(view=False, get_real=True, normalize=True)
    im3 = model.sample(key, get_real=True, normalize=True)
    im4 = model.sample(key, view=False, get_real=True, normalize=True)
    im5 = irfftn(
        model.render(get_real=False, normalize=True),
        s=model.scattering.manager.shape,
    )
    im6 = irfftn(
        model.render(get_real=False, normalize=True, view=False),
        s=model.scattering.manager.padded_shape,
    )
    for im in [im1, im2, im3, im4, im5, im6]:
        np.testing.assert_allclose(jnp.std(im), jnp.asarray(1.0), rtol=1e-3)
        np.testing.assert_allclose(jnp.mean(im), jnp.asarray(0.0), atol=1e-12)
