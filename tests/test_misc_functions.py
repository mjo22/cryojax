import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import cryojax.image as cxi


jax.config.update("jax_enable_x64", True)


#
# Downsampling
#
@pytest.mark.parametrize(
    "shape, downsample_factor",
    (((10, 10), 2), ((11, 11), 2)),
)
def test_downsample_preserves_sum(shape, downsample_factor):
    upsampled_shape = tuple(downsample_factor * s for s in shape)
    rng_key = jr.PRNGKey(seed=1234)
    upsampled_image = 2.0 + 1.0 * jr.normal(rng_key, upsampled_shape)
    image = cxi.downsample_with_fourier_cropping(upsampled_image, downsample_factor)
    np.testing.assert_allclose(image.sum(), upsampled_image.sum())


#
# Normalization
#
def test_fourier_vs_real_normalize(noisy_model):
    key = jax.random.key(1234)
    im1 = cxi.normalize_image(
        noisy_model.render(key, outputs_real_space=True),
        input_is_real_space=True,
    )
    im2 = cxi.irfftn(
        cxi.normalize_image(
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


#
# FFT
#
@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 10), (11, 11), (11, 11, 11)])
def test_fft_agrees_with_jax_numpy(shape):
    random = jnp.asarray(np.random.randn(*shape))
    # fftn
    np.testing.assert_allclose(random, cxi.ifftn(cxi.fftn(random)).real)
    np.testing.assert_allclose(
        cxi.ifftn(cxi.fftn(random)).real, jnp.fft.ifftn(jnp.fft.fftn(random)).real
    )
    # rfftn
    np.testing.assert_allclose(random, cxi.irfftn(cxi.rfftn(random), s=shape))
    np.testing.assert_allclose(
        cxi.irfftn(cxi.rfftn(random), s=shape),
        jnp.fft.irfftn(jnp.fft.rfftn(random), s=shape),
    )
