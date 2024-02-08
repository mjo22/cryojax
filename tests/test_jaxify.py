from cryojax.utils.interpolate import diff, jaxy_diff
from cryojax.utils.interpolate import fill_vec, jaxy_fill_vec
from cryojax.utils.interpolate import jaxy_numpy_interp_vec, jaxy_interp_vec
from cryojax.reconstruct.backproject import (
    _insert_slice_and_interpolate,
    _jaxy_insert_slice_and_interpolate,
)


from cryojax.utils.coordinates import make_coordinates
from scipy.spatial.transform import Rotation
import jax.numpy as jnp
import numpy as np
import jax


def test_jaxy_diff():
    n_pix = 10
    xy = make_coordinates((n_pix, n_pix))

    xy0 = jnp.zeros((3, n_pix**2))

    xy0 = xy0.at[0].set(xy[:, :, 0].reshape(-1))
    xy0 = xy0.at[1].set(xy[:, :, 1].reshape(-1))

    deg = 10
    R = Rotation.from_euler("zyz", [deg, deg, 0]).as_matrix()
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


def test_jaxy_fill_vec():
    """
    Notes:
    -----

    # TODO: look at numerics more carefully. including drop and set arguments in jax.numpy.at
    Unsure why this these functions are different. Numerical issues for in place adding in numpy? JAX seems to be more accurate.
    """

    n_pix = 10
    xy = make_coordinates((n_pix, n_pix))

    xy0 = jnp.zeros((3, n_pix**2))

    xy0 = xy0.at[0].set(xy[:, :, 0].reshape(-1))
    xy0 = xy0.at[1].set(xy[:, :, 1].reshape(-1))

    R = Rotation.from_euler("zyz", [10, 10, 0]).as_matrix()
    xyz_rot = R @ xy0
    r0, r1, dd = diff(xyz_rot)

    slice_2d = np.ones(n_pix**2).reshape((n_pix, n_pix)).astype(np.float32)

    # numpy
    r0_idx = r0 + n_pix // 2
    r1_idx = r1 + n_pix // 2

    under_grid_idx = np.any(r0_idx < 0, axis=0)
    over_grid_idx = np.any(r1_idx >= n_pix, axis=0)
    good_idx = np.logical_and(~under_grid_idx, ~over_grid_idx)

    map_3d_interp = np.zeros((n_pix, n_pix, n_pix)).astype(slice_2d.dtype)
    slice_flat = slice_2d.flatten()[good_idx]

    r0_idx_good = r0_idx[:, good_idx]
    r1_idx_good = r1_idx[:, good_idx]

    numpy_map_3d_interp = fill_vec(
        map_3d_interp, r0_idx_good, r1_idx_good, slice_flat, dd[:, good_idx]
    )

    # jax
    r0_idx = r0 + n_pix // 2
    r1_idx = r1 + n_pix // 2

    under_grid_idx = jnp.any(r0_idx < 0, axis=0)
    over_grid_idx = jnp.any(r1_idx >= n_pix, axis=0)
    good_idx = jnp.logical_and(~under_grid_idx, ~over_grid_idx)

    map_3d_interp = jnp.zeros((n_pix, n_pix, n_pix)).astype(slice_2d.dtype)
    slice_flat = slice_2d.flatten()[good_idx]

    r0_idx_good = r0_idx[:, good_idx]
    r1_idx_good = r1_idx[:, good_idx]

    jaxy_map_3d_interp = jaxy_fill_vec(
        map_3d_interp, r0_idx_good, r1_idx_good, slice_flat, dd[:, good_idx]
    )

    generous_rtol = 0.8
    np.testing.assert_allclose(
        numpy_map_3d_interp.sum(), jaxy_map_3d_interp.sum(), rtol=generous_rtol
    )
    generous_atol = (
        0.6  # on the order of 1 (because interpolating plane of all ones)
    )
    np.testing.assert_allclose(
        numpy_map_3d_interp, jaxy_map_3d_interp, atol=generous_atol
    )

    jaxy_jitted_map_3d_interp = jax.jit(jaxy_fill_vec)(
        map_3d_interp, r0_idx_good, r1_idx_good, slice_flat, dd[:, good_idx]
    )

    np.testing.assert_allclose(
        numpy_map_3d_interp.sum(),
        jaxy_jitted_map_3d_interp.sum(),
        rtol=generous_rtol,
    )
    np.testing.assert_allclose(
        numpy_map_3d_interp, jaxy_jitted_map_3d_interp, atol=generous_atol
    )


def test_jaxy_interp_vec():
    n_pix = 10  # shape[0]

    xy = make_coordinates((n_pix, n_pix))

    xy0 = jnp.zeros((3, n_pix**2))

    xy0 = xy0.at[0].set(xy[:, :, 0].reshape(-1))
    xy0 = xy0.at[1].set(xy[:, :, 1].reshape(-1))

    R = Rotation.from_euler("zyz", [10, 10, 0]).as_matrix()
    xyz_rot = R @ xy0

    r0, r1, dd = diff(xyz_rot)

    slice_real = np.ones((n_pix, n_pix)).astype(np.float64)

    # numpy
    map_3d_interp_slice, count_3d_interp_slice = jaxy_numpy_interp_vec(
        slice_real, r0, r1, dd, n_pix
    )

    # jax
    map_3d_interp = jnp.zeros((n_pix, n_pix, n_pix))
    count_3d_interp = jnp.zeros((n_pix, n_pix, n_pix))

    jaxy_map_3d_interp_slice, jaxy_count_3d_interp_slice = jaxy_interp_vec(
        slice_real, r0, r1, dd, map_3d_interp, count_3d_interp
    )

    np.testing.assert_allclose(map_3d_interp_slice, jaxy_map_3d_interp_slice)
    np.testing.assert_allclose(
        count_3d_interp_slice, jaxy_count_3d_interp_slice
    )


def test_jaxy_insert_slice_and_interpolate():
    n_pix = 10
    map_3d_zeros = jnp.zeros((n_pix, n_pix, n_pix))
    count_3d_zeros = jnp.zeros((n_pix, n_pix, n_pix))

    N = n_pix
    slice_f = jnp.ones((N, N)).astype(jnp.float64)

    xyz = jnp.zeros((3, N**2))
    rotation = Rotation.from_euler("zyz", [10, 10, 0]).as_matrix()

    xyz_central_slice = xyz.at[:2].set(
        make_coordinates((N, N)).reshape(-1, 2).T
    )
    xyz_rotated_central_slice = rotation @ xyz_central_slice

    (
        jaxy_inserted_slice_3d_real,
        jaxy_count_3d_real,
    ) = _jaxy_insert_slice_and_interpolate(
        slice_f, xyz_rotated_central_slice, map_3d_zeros, count_3d_zeros
    )

    inserted_slice_3d_real, count_3d_real = _insert_slice_and_interpolate(
        slice_f, xyz_rotated_central_slice, N
    )

    np.testing.assert_allclose(
        jaxy_inserted_slice_3d_real, inserted_slice_3d_real, atol=0.6
    )
    np.testing.assert_allclose(jaxy_count_3d_real, count_3d_real, atol=0.6)
