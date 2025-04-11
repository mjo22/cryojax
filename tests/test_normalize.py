import jax
import jax.numpy as jnp
import numpy as np

from cryojax.image import irfftn, normalize_image


jax.config.update("jax_enable_x64", True)


def test_fourier_vs_real_normalized_image(noisy_model):
    key = jax.random.key(1234)
    im1 = normalize_image(
        noisy_model.render(key, outputs_real_space=True), input_is_real_space=True
    )
    im2 = irfftn(
        normalize_image(
            noisy_model.render(outputs_real_space=False),
            input_is_real_space=False,
            input_is_rfft=True,
            shape_in_real_space=im1.shape,
        ),
        s=noisy_model.instrument_config.shape,
    )  # type: ignore
    for im in [im1, im2]:
        np.testing.assert_allclose(jnp.std(im), jnp.asarray(1.0), rtol=1e-3)
        np.testing.assert_allclose(jnp.mean(im), jnp.asarray(0.0), atol=1e-8)
