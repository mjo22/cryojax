#!/usr/bin/env python3
"""
Routines defining coordinate system for 3D templates or images.
"""

__all__ = ["coordinatize", "radial_grid"]

import jax.numpy as jnp
import numpy as np
from typing import Optional, Sequence
from jax_2dtm.types import Array, ArrayLike


def coordinatize(volume: ArrayLike, pixel_size: Optional[float] = 1.0) -> Array:
    """
    Returns flattened coordinate system and 3D volume or 2D image
    of shape ``(N, ndim)``, where ``ndim = volume.ndim`` and
    ``N = N1*N2*N3`` or ``N = N2*N3``. The coordinate system is in
    pixel coordinates with zero in the center.

    Parameters
    ----------
    volume : `jnp.ndarray`, shape `(N1, N2, N3)` or `(N2, N3)`
        3D volume or 2D image.
    pixel_size : `float`, optional
        Camera pixel size.

    Returns
    -------
    flat : `jnp.ndarray`, shape `(N, ndim)`
        Flattened volume or image.
    coords : `jnp.ndarray`, shape `(N, ndim)`
        Flattened cartesian coordinate system.
    """
    ndim, shape = volume.ndim, volume.shape
    N = np.prod(shape)
    coords = jnp.zeros((N, ndim), dtype=int)

    # Generate cubic grid and fill coordinate array
    R = radial_grid(shape)
    for i in range(ndim):
        coords = coords.at[:, i].set(pixel_size * R[i].ravel())

    # Flattened template
    flat = volume.ravel()

    return flat, coords


def radial_grid(shape: Sequence[int]) -> Array:
    """
    Create a radial coordinate system on a grid.
    This can be used for real and fourier space
    calculations. If used for fourier space, the
    zero-frequency component is in the center.

    Arguments
    ---------

    Returns
    -------

    """
    ndim = len(shape)
    rcoords1D = []
    for i in range(ndim):
        ni = shape[i]
        ri = jnp.fft.fftshift(jnp.fft.fftfreq(ni))*ni
        rcoords1D.append(ri)

    rcoords = jnp.meshgrid(*rcoords1D, indexing='ij')

    return rcoords


if __name__ == '__main__':
    from jax_2dtm.io import load_mrc

    template = load_mrc("./example/6dpu_14pf_bfm1_ps1_1.mrc")
    shape = template.shape

    flat, coords = coordinatize(template)
