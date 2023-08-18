from .test_pipeline import setup, scattering_model

import jax.numpy as jnp
import numpy as np
from cryojax.utils import fft, irfft


def test_fft(scattering_model):
    image = fft(scattering_model())
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
    # Run tests with cryojax.utils and random data
    np.testing.assert_allclose(
        irfft(fft(random)), jnp.fft.ifftn(jnp.fft.fftn(random)).real, **rkwargs
    )
    np.testing.assert_allclose(random, irfft(fft(random)), **rkwargs)
    np.testing.assert_allclose(fft(random), fft(irfft(fft(random))), **fkwargs)
    # Run tests with an image
    np.testing.assert_allclose(
        irfft(image), irfft(fft(irfft(image))), **rkwargs
    )
    np.testing.assert_allclose(image, fft(irfft(image)), **fkwargs)
