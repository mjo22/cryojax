from cryojax.utils.interpolate import jaxy_diff, diff
from cryojax.utils.coordinates import make_coordinates
from scipy.spatial.transform import Rotation
import jax.numpy as jnp
import numpy as np


def test_jaxy_diff():
    xy = make_coordinates((10, 10))

    xy0 = jnp.zeros((3, 10**2))

    xy0 = xy0.at[0].set(xy[:, :, 0].reshape(-1))
    xy0 = xy0.at[1].set(xy[:, :, 1].reshape(-1))

    R = Rotation.from_euler("zyz", [10, 10, 0]).as_matrix()
    xyz_rot = R @ xy0

    r0, r1, dd = diff(xyz_rot)
    jaxy_r0, jaxy_r1, jaxy_dd = jaxy_diff(xyz_rot)

    np.testing.assert_allclose(r0, jaxy_r0)
    np.testing.assert_allclose(r1, jaxy_r1)
    np.testing.assert_allclose(dd, jaxy_dd)

    jaxy_jitted_r0, jaxy_jitted_r1, jaxy_jitted_dd = jax.jit(jaxy_diff)(
        xyz_rot
    )

    np.testing.assert_allclose(jaxy_jitted_r0, jaxy_r0)
    np.testing.assert_allclose(jaxy_jitted_r1, jaxy_r1)
    np.testing.assert_allclose(jaxy_jitted_dd, jaxy_dd)
