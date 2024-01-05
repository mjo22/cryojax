"""
Interpolation routines.
"""

__all__ = ["resize", "scale", "scale_and_translate", "map_coordinates"]

from typing import Union, Any
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates as _map_coordinates
from jax.image import resize as _resize
from jax.image import scale_and_translate as _scale_and_translate

import numpy as np


def resize(
    image: Array,
    shape: tuple[int, int],
    method="lanczos5",
    align_corners: bool = True,
    **kwargs: Any
) -> Array:
    """
    Resize an image with interpolation.

    Wraps ``jax.image.resize``.
    """
    if align_corners:
        return _resize_with_aligned_corners(image, shape, method, **kwargs)
    else:
        return _resize(image, shape, method, **kwargs)


def scale_and_translate(
    image: Array,
    shape: tuple[int, int],
    scale: Float[Array, "2"],
    translation: Float[Array, "2"],
    method="lanczos5",
    **kwargs
) -> Array:
    """
    Resize, scale, and translate an image with interpolation.

    Wraps ``jax.image.scale_and_translate``.
    """
    image = jnp.asarray(image)
    spatial_dims = (0, 1)
    N1, N2 = image.shape
    translation += (1 - scale) * jnp.array([N2 // 2, N1 // 2], dtype=float)
    return _scale_and_translate(
        image, shape, spatial_dims, scale, translation, method, **kwargs
    )


def scale(
    image: Array,
    shape: tuple[int, int],
    scale: Float[Array, "2"],
    method="lanczos5",
    **kwargs
) -> Array:
    """
    Resize and scale an image with interpolation.

    Wraps ``jax.image.scale_and_translate``.
    """
    translation = jnp.array([0.0, 0.0])
    return scale_and_translate(
        image, shape, scale, translation, method=method, **kwargs
    )


def map_coordinates(
    input: Array, coordinates: Array, order=1, mode="wrap", cval=0.0
) -> Array:
    """
    Interpolate a set of points in fourier space on a grid
    with a given coordinate system onto a new coordinate system.
    """
    input, coordinates = jnp.asarray(input), jnp.asarray(coordinates)
    N1, N2, N3 = input.shape
    box_shape = jnp.array([N1, N2, N3], dtype=float)[:, None, None, None]
    coordinates = jnp.transpose(coordinates, axes=[3, 0, 1, 2])
    # Flip negative valued frequencies to get the logical coordinates.
    coordinates = jnp.where(
        coordinates < 0, box_shape + coordinates, coordinates
    )
    return _map_coordinates(input, coordinates, order, mode=mode, cval=cval)


def _resize_with_aligned_corners(
    image: Array,
    shape: tuple[int, ...],
    method: Union[str, jax.image.ResizeMethod],
    antialias: bool = False,
) -> Array:
    """
    Alternative to jax.image.resize(), which emulates
    align_corners=True in PyTorch's interpolation functions.

    Adapted from https://github.com/google/jax/issues/11206.
    ."""
    image = jnp.asarray(image)
    spatial_dims = tuple(
        i
        for i in range(len(shape))
        if not jax.core.symbolic_equal_dim(image.shape[i], shape[i])
    )
    scale = jnp.array(
        [(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims]
    )
    translation = -(scale / 2.0 - 0.5)
    return _scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )

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