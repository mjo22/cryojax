"""
Interpolation routines.
"""

__all__ = ["interp_vec", "diff", "fill_vec"]

import numpy as np
import jax.numpy as jnp
import jax


def diff(xyz_rot):
    """Precompute for linear interpolation.

    Parameters
    ----------
    xyz_rot : array
        Rotated plane
        Shape (3,n_pix**2)

    Returns
    -------
    r0,r1 : array
        Shape (3,n_pix**2)
        Location to nearby grid points (r0 low, r1 high)
    dd : array
        Shape (8,n_pix**2)
        Distance to 8 nearby voxels. Linear interpolation kernel.
    """
    r0 = np.floor(xyz_rot).astype(int)
    r1 = r0 + 1
    fr = xyz_rot - r0
    mfr = 1 - fr
    mfx, mfy, mfz = mfr[0], mfr[1], mfr[-1]
    fx, fy, fz = fr[0], fr[1], fr[-1]
    dd000 = mfz * mfy * mfx
    dd001 = mfz * mfy * fx
    dd010 = mfz * fy * mfx
    dd011 = mfz * fy * fx
    dd100 = fz * mfy * mfx
    dd101 = fz * mfy * fx
    dd110 = fz * fy * mfx
    dd111 = fz * fy * fx
    dd = np.array([dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111])
    return r0, r1, dd


def jaxy_diff(xyz_rot):
    """Precompute for linear interpolation.

    Parameters
    ----------
    xyz_rot : array
        Rotated plane
        Shape (3,n_pix**2)

    Returns
    -------
    r0,r1 : array
        Shape (3,n_pix**2)
        Location to nearby grid points (r0 low, r1 high)
    dd : array
        Shape (8,n_pix**2)
        Distance to 8 nearby voxels. Linear interpolation kernel.
    """
    r0 = jnp.floor(xyz_rot).astype(int)
    r1 = r0 + 1
    fr = xyz_rot - r0
    mfr = 1 - fr
    mfx, mfy, mfz = mfr[0], mfr[1], mfr[-1]
    fx, fy, fz = fr[0], fr[1], fr[-1]
    dd000 = mfz * mfy * mfx
    dd001 = mfz * mfy * fx
    dd010 = mfz * fy * mfx
    dd011 = mfz * fy * fx
    dd100 = fz * mfy * mfx
    dd101 = fz * mfy * fx
    dd110 = fz * fy * mfx
    dd111 = fz * fy * fx
    dd = jnp.array([dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111])
    return r0, r1, dd


def fill_vec(map_3d_interp, r0_idx_good, r1_idx_good, slice_flat_good, dd):
    """Linear interpolation kernel.

    Interpolates in nearby 8 voxels.
    Done on good_idx indices in domain: len(good_idx)<=n_pix**2

    Parameters
    ----------
    map_3d_interp : array
        Empty flattened 3D map to be interpolated into/onto
    r0_idx_good, r1_idx_good
        Shape (3,len(good_idx))
        Location to nearby grid points (r0 low, r1 high)
    slice_flat_good : array
        Shape (len(good_idx),)
        Flattened 2D slice
    dd : array
        Shape (8,len(good_idx))
        Distance to 8 nearby voxels. Linear interpolation kernel.

    Returns
    -------
    map_3d_interp : array
        Filled flattened 3D map to be interpolated into/onto
    """
    dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111 = dd

    map_3d_interp[r0_idx_good[0], r0_idx_good[1], r0_idx_good[-1]] += (
        slice_flat_good * dd000
    )
    map_3d_interp[r1_idx_good[0], r0_idx_good[1], r0_idx_good[-1]] += (
        slice_flat_good * dd001
    )
    map_3d_interp[r0_idx_good[0], r1_idx_good[1], r0_idx_good[-1]] += (
        slice_flat_good * dd010
    )
    map_3d_interp[r1_idx_good[0], r1_idx_good[1], r0_idx_good[-1]] += (
        slice_flat_good * dd011
    )
    map_3d_interp[r0_idx_good[0], r0_idx_good[1], r1_idx_good[-1]] += (
        slice_flat_good * dd100
    )
    map_3d_interp[r1_idx_good[0], r0_idx_good[1], r1_idx_good[-1]] += (
        slice_flat_good * dd101
    )
    map_3d_interp[r0_idx_good[0], r1_idx_good[1], r1_idx_good[-1]] += (
        slice_flat_good * dd110
    )
    map_3d_interp[r1_idx_good[0], r1_idx_good[1], r1_idx_good[-1]] += (
        slice_flat_good * dd111
    )
    return map_3d_interp


def jaxy_fill_vec(
    map_3d_interp, r0_idx_good, r1_idx_good, slice_flat_good, dd
):
    """Linear interpolation kernel.

    Interpolates in nearby 8 voxels.
    Done on good_idx indices in domain: len(good_idx)<=n_pix**2

    Parameters
    ----------
    map_3d_interp : array
        Empty flattened 3D map to be interpolated into/onto
    r0_idx_good, r1_idx_good
        Shape (3,len(good_idx))
        Location to nearby grid points (r0 low, r1 high)
    slice_flat_good : array
        Shape (len(good_idx),)
        Flattened 2D slice
    dd : array
        Shape (8,len(good_idx))
        Distance to 8 nearby voxels. Linear interpolation kernel.

    Returns
    -------
    map_3d_interp : array
        Filled flattened 3D map to be interpolated into/onto
    """
    dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111 = dd

    map_3d_interp = map_3d_interp.at[
        r0_idx_good[0], r0_idx_good[1], r0_idx_good[-1]
    ].add(slice_flat_good * dd000)
    map_3d_interp = map_3d_interp.at[
        r1_idx_good[0], r0_idx_good[1], r0_idx_good[-1]
    ].add(slice_flat_good * dd001)
    map_3d_interp = map_3d_interp.at[
        r0_idx_good[0], r1_idx_good[1], r0_idx_good[-1]
    ].add(slice_flat_good * dd010)
    map_3d_interp = map_3d_interp.at[
        r1_idx_good[0], r1_idx_good[1], r0_idx_good[-1]
    ].add(slice_flat_good * dd011)
    map_3d_interp = map_3d_interp.at[
        r0_idx_good[0], r0_idx_good[1], r1_idx_good[-1]
    ].add(slice_flat_good * dd100)
    map_3d_interp = map_3d_interp.at[
        r1_idx_good[0], r0_idx_good[1], r1_idx_good[-1]
    ].add(slice_flat_good * dd101)
    map_3d_interp = map_3d_interp.at[
        r0_idx_good[0], r1_idx_good[1], r1_idx_good[-1]
    ].add(slice_flat_good * dd110)
    map_3d_interp = map_3d_interp.at[
        r1_idx_good[0], r1_idx_good[1], r1_idx_good[-1]
    ].add(slice_flat_good * dd111)

    return map_3d_interp


def interp_vec(slice_2d, r0, r1, dd, n_pix):
    """Linear interpolation.

    Parameters
    ----------
    slice_2d : array
        Slice to be interpolated
        Shape (n_pix,n_pix)

    Returns
    -------
    r0,r1 : array
        Shape (3,n_pix**2)
        Location to nearby grid points (r0 low, r1 high)
    dd : array
        Shape (8,n_pix**2)
        Distance to 8 nearby voxels. Linear interpolation kernel.
    """
    r0_idx = r0 + n_pix // 2
    r1_idx = r1 + n_pix // 2

    under_grid_idx = np.any(r0_idx < 0, axis=0)
    over_grid_idx = np.any(r1_idx >= n_pix, axis=0)
    good_idx = np.logical_and(~under_grid_idx, ~over_grid_idx)

    map_3d_interp = np.zeros((n_pix, n_pix, n_pix)).astype(slice_2d.dtype)
    count_3d_interp = np.zeros((n_pix, n_pix, n_pix))
    ones = np.ones(n_pix * n_pix)[good_idx]
    slice_flat = slice_2d.flatten()[good_idx]

    r0_idx_good = r0_idx[:, good_idx]
    r1_idx_good = r1_idx[:, good_idx]

    map_3d_interp = fill_vec(
        map_3d_interp, r0_idx_good, r1_idx_good, slice_flat, dd[:, good_idx]
    )
    count_3d_interp = fill_vec(
        count_3d_interp, r0_idx_good, r1_idx_good, ones, dd[:, good_idx]
    )

    return map_3d_interp, count_3d_interp


def jaxy_numpy_interp_vec(slice_2d, r0, r1, dd, n_pix):
    """Linear interpolation.

    Parameters
    ----------
    slice_2d : array
        Slice to be interpolated
        Shape (n_pix,n_pix)

    Returns
    -------
    r0,r1 : array
        Shape (3,n_pix**2)
        Location to nearby grid points (r0 low, r1 high)
    dd : array
        Shape (8,n_pix**2)
        Distance to 8 nearby voxels. Linear interpolation kernel.
    """
    r0_idx = r0 + n_pix // 2
    r1_idx = r1 + n_pix // 2

    under_grid_idx = jnp.any(r0_idx < 0, axis=0)
    over_grid_idx = jnp.any(r1_idx >= n_pix, axis=0)
    good_idx = jnp.logical_and(~under_grid_idx, ~over_grid_idx)

    map_3d_interp = jnp.zeros((n_pix, n_pix, n_pix)).astype(slice_2d.dtype)
    count_3d_interp = jnp.zeros((n_pix, n_pix, n_pix))
    ones = jnp.ones(n_pix * n_pix)[good_idx]
    slice_flat = slice_2d.flatten()[good_idx]

    r0_idx_good = r0_idx[:, good_idx]
    r1_idx_good = r1_idx[:, good_idx]

    map_3d_interp = jaxy_fill_vec(
        map_3d_interp, r0_idx_good, r1_idx_good, slice_flat, dd[:, good_idx]
    )
    count_3d_interp = jaxy_fill_vec(
        count_3d_interp, r0_idx_good, r1_idx_good, ones, dd[:, good_idx]
    )

    return map_3d_interp, count_3d_interp


def jaxy_interp_vec(slice_2d, r0, r1, dd, map_3d_interp, count_3d_interp):
    """Linear interpolation.

    Parameters
    ----------
    slice_2d : array
        Slice to be interpolated
        Shape (n_pix,n_pix)

    Returns
    -------
    r0,r1 : array
        Shape (3,n_pix**2)
        Location to nearby grid points (r0 low, r1 high)
    dd : array
        Shape (8,n_pix**2)
        Distance to 8 nearby voxels. Linear interpolation kernel.
    """
    n_pix = len(map_3d_interp)
    r0_idx = r0 + n_pix // 2
    r1_idx = r1 + n_pix // 2

    under_grid_idx = jnp.any(r0_idx < 0, axis=0)
    over_grid_idx = jnp.any(r1_idx >= n_pix, axis=0)
    good_idx = jnp.logical_and(~under_grid_idx, ~over_grid_idx)

    fill_value = 0  # important for dd so that nothing added in that is not supposed to be
    r0_idx_good_dense_filled = jnp.where(good_idx, r0_idx, fill_value)
    r1_idx_good_dense_filled = jnp.where(good_idx, r1_idx, fill_value)
    dd_dense_filled = jnp.where(good_idx, dd, fill_value)

    ones_dense_filled = jnp.where(
        good_idx, jnp.ones(n_pix * n_pix).flatten(), fill_value
    )
    slice_flat_dense_filled = jnp.where(
        good_idx, slice_2d.flatten(), fill_value
    )

    map_3d_interp = jaxy_fill_vec(
        map_3d_interp,
        r0_idx_good_dense_filled,
        r1_idx_good_dense_filled,
        slice_flat_dense_filled,
        dd_dense_filled,
    )
    count_3d_interp = jaxy_fill_vec(
        count_3d_interp,
        r0_idx_good_dense_filled,
        r1_idx_good_dense_filled,
        ones_dense_filled,
        dd_dense_filled,
    )

    return map_3d_interp, count_3d_interp
