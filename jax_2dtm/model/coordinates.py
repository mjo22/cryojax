#!/usr/bin/env python3
"""
Routines defining coordinate system for 3D templates or images.
"""

import jax.numpy as jnp
import numpy as np


def coordinatize(volume):
    """
    Returns flattened coordinate system and 3D volume or 2D image
    of shape ``(N, ndim)``, where ``ndim = volume.ndim`` and
    ``N = N1*N2*N3`` or ``N = N2*N3``. The coordinate system is in
    pixel coordinates with zero in the center.

    Parameters
    ----------
    volume : `jnp.ndarray`, shape `(N1, N2, N3)` or `(N2, N3)`
        3D volume or 2D image.

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

    # Fill coordinate array
    r = [jnp.arange(-Ni//2, Ni//2) for Ni in shape]
    R = jnp.meshgrid(*r, indexing="ij")
    for i in range(ndim):
        coords = coords.at[:, i].set(R[i].ravel())

    # Flattened template
    flat = volume.ravel()

    return flat, coords


if __name__ == '__main__':
    from template import read_mrc

    template = read_mrc("./example/6dpu_14pf_bfm1_ps1_1.mrc")
    shape = template.shape

    flat, coords = coordinatize(template)
