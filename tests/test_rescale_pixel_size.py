import jax.random as jr
import numpy as np
import pytest

from cryojax.image import downsample_with_fourier_cropping


@pytest.mark.parametrize("downsampling_factor", [1.0, 2.0, 4.0])
def test_rescale_by_downsampling_preserves_sum(downsampling_factor):
    rng_key = jr.PRNGKey(seed=1234)
    random = 2.0 + 1.0 * jr.normal(rng_key, (100, 100))
    downsampled_random = downsample_with_fourier_cropping(random, downsampling_factor)
    np.testing.assert_allclose(random.sum(), downsampled_random.sum())
