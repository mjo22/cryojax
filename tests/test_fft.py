import pytest

import jax.numpy as jnp
import numpy as np
from cryojax.utils import fftn, ifftn


@pytest.mark.parametrize(
    "model", ["noisy_model", "maskless_model", "likelihood_model"]
)
def test_fft(model, request):
    model = request.getfixturevalue(model)
    image = fftn(model.render())
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
        ifftn(fftn(random)).real,
        jnp.fft.ifftn(jnp.fft.fftn(random)).real,
        **rkwargs
    )
    np.testing.assert_allclose(random, ifftn(fftn(random)).real, **rkwargs)
    np.testing.assert_allclose(
        fftn(random), fftn(ifftn(fftn(random))), **fkwargs
    )
    # Run tests with an image
    np.testing.assert_allclose(
        ifftn(image).real, ifftn(fftn(ifftn(image))).real, **rkwargs
    )
    np.testing.assert_allclose(image, fftn(ifftn(image).real), **fkwargs)
