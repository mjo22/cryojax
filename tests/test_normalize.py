import pytest

import jax
import jax.numpy as jnp
from cryojax.utils import irfftn


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
    for im in [im1, im2, im3, im4]:
        assert pytest.approx(jnp.std(im).item()) == 1.0
        assert pytest.approx(jnp.mean(im).item()) == 0.0
