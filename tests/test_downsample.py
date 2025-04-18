import jax.numpy as jnp
import numpy as np
import pytest

from cryojax.image import downsample_with_fourier_cropping


@pytest.mark.parametrize(
    "shape, downsample_factor",
    (((10, 10), 2), ((11, 11), 2)),
)
def test_downsample_preserves_sum(shape, downsample_factor):
    upsampled_shape = tuple(downsample_factor * s for s in shape)
    upsampled_image = jnp.ones(upsampled_shape)
    image = downsample_with_fourier_cropping(upsampled_image, downsample_factor)
    np.testing.assert_equal(np.sum(image), np.sum(upsampled_image))
