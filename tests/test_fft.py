from .test_pipeline import model

import jax.numpy as jnp
import numpy as np
from jax_2dtm.utils import fft, ifft


def test_fft(model):
    image = model()
    random = jnp.asarray(np.random.randn(*image.shape))
    # Set tolerance based on tests with jnp.fft + random data
    rkwargs = dict(zip(["atol", "rtol"], [1e-6, 5e-4]))
    fkwargs = dict(zip(["atol", "rtol"], [5e-5, 1e-6]))
    # Run tests with jnp.fft and random data
    np.testing.assert_allclose(
        random, jnp.fft.ifftn(jnp.fft.fftn(random)).real, **rkwargs
    )
    np.testing.assert_allclose(
        jnp.fft.fftn(random),
        jnp.fft.fftn(jnp.fft.ifftn(jnp.fft.fftn(random)).real),
        **fkwargs
    )
    # Run tests with jax_2dtm.utils and random data
    np.testing.assert_allclose(
        ifft(fft(random)), jnp.fft.ifftn(jnp.fft.fftn(random)).real, **rkwargs
    )
    np.testing.assert_allclose(random, ifft(fft(random)), **rkwargs)
    np.testing.assert_allclose(fft(random), fft(ifft(fft(random))), **fkwargs)
    # Run tests with an image
    np.testing.assert_allclose(ifft(image), ifft(fft(ifft(image))), **rkwargs)
    np.testing.assert_allclose(image, fft(ifft(image)), **fkwargs)
