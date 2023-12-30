import pytest

import jax.numpy as jnp
import numpy as np
from cryojax.utils import fftn, ifftn
from jax import config

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["noisy_model", "noiseless_model"])
def test_fft(model, request):
    model = request.getfixturevalue(model)
    image = model.render()
    random = jnp.asarray(np.random.randn(*image.shape))
    # Set tolerance based on tests with jnp.fft + random data
    rkwargs = dict(zip(["atol", "rtol"], [0, 1e-7]))
    fkwargs = dict(zip(["atol", "rtol"], [0, 1e-7]))
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
        ifftn(fftn(random)).real,
        jnp.fft.ifftn(jnp.fft.fftn(random)).real,
        **rkwargs
    )
    np.testing.assert_allclose(random, ifftn(fftn(random)).real, **rkwargs)
    np.testing.assert_allclose(
        fftn(random), fftn(ifftn(fftn(random))), **fkwargs
    )
    # Run tests with an image
    np.testing.assert_allclose(image, ifftn(fftn(image)).real, **rkwargs)
    np.testing.assert_allclose(
        fftn(image), fftn(ifftn(fftn(image)).real), **fkwargs
    )
