import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cryojax.image import irfftn


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "model",
    [
        "noisy_model",
        "noiseless_model",
    ],
)
def test_normalized_pipeline_image(model, request):
    model = request.getfixturevalue(model)
    key = jax.random.PRNGKey(1234)
    im1 = model.render(get_real=True, normalize=True)
    im2 = model.sample(key, get_real=True, normalize=True)
    im3 = irfftn(
        model.render(get_real=False, normalize=True),
        s=model.config.shape,
    )
    im4 = irfftn(
        model.render(get_real=False, normalize=True),
        s=model.config.shape,
    )
    for im in [im1, im2, im3, im4]:
        np.testing.assert_allclose(jnp.std(im), jnp.asarray(1.0), rtol=1e-3)
        np.testing.assert_allclose(jnp.mean(im), jnp.asarray(0.0), atol=1e-8)
