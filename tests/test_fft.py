import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cryojax.image import fftn, ifftn


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["noisy_model", "noiseless_model"])
def test_fft(model, request):
    model = request.getfixturevalue(model)
    image = model.render()
    random = jnp.asarray(np.random.randn(*image.shape))
    # Run tests with jnp.fft and random data
    np.testing.assert_allclose(random, jnp.fft.ifftn(jnp.fft.fftn(random)).real)
    np.testing.assert_allclose(
        jnp.fft.fftn(random),
        jnp.fft.fftn(jnp.fft.ifftn(jnp.fft.fftn(random)).real),
    )
    # Run tests with cryojax.image and random data
    np.testing.assert_allclose(
        ifftn(fftn(random)).real, jnp.fft.ifftn(jnp.fft.fftn(random)).real
    )
    np.testing.assert_allclose(random, ifftn(fftn(random)).real)
    np.testing.assert_allclose(fftn(random), fftn(ifftn(fftn(random))))
    # Run tests with an image
    np.testing.assert_allclose(image, ifftn(fftn(image)).real)
    # ... test zero mode separately
    np.testing.assert_allclose(
        fftn(image)[1:, 1:], fftn(ifftn(fftn(image)).real)[1:, 1:], atol=1e-10
    )
    np.testing.assert_allclose(
        fftn(image)[0, 0], fftn(ifftn(fftn(image)).real)[0, 0], atol=1e-12
    )
