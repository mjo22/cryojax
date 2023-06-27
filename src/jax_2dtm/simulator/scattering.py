#!/usr/bin/env python3
"""
Routines to model image formation
"""

import jax.numpy as jnp
from functools import partial
from typing import Tuple
from jax import jit
from jax_2dtm.types import Array


@partial(jit, static_argnames=['shape'])
def project(volume: Array, coords: Array, shape: Tuple[int]) -> Array:
    """
    Project 3D volume onto imaging plane
    by computing a histogram.

    Parameters
    ----------
    volume : shape `(N, 3)`
        3D volume.
    coords : shape `(N, 3)`
        Coordinate system.
    shape :
        A tuple denoting the shape of the output image, given
        by ``(N2, N3)``
    Returns
    -------
    projection : shape `(N2, N3)`
        Projection of volume onto imaging plane,
        which is taken to be axis 0.
    """
    N2, N3 = shape[0], shape[1]
    # Round coordinates for binning
    rounded_coords = jnp.rint(coords).astype(int)
    # Shift coordinates back to zero in the corner, rather than center
    y_coords, z_coords = rounded_coords[:, 1]+N2//2, rounded_coords[:, 2]+N3//2
    # Bin values on the same y-z plane
    flat_coords = jnp.ravel_multi_index((y_coords, z_coords), (N2, N3), mode='clip')
    projection = jnp.bincount(flat_coords, weights=volume, length=N2*N3).reshape((N2, N3))

    return projection


def normalize(image):
    """
    Normalize image.

    Parameters
    ----------


    Returns
    -------

    """
    pass


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from jax import jit
    from jax_2dtm.io import load_volume
    from .rotations import rotate_rpy
    from .interpolation import interpolate
    from .coordinates import coordinatize

    template = load_volume("./example/6dpu_14pf_bfm1_ps1_1.mrc")
    shape = template.shape[1:]

    # Read coordinates
    model, coords = coordinatize(template)

    # Apply rotation
    rpy = jnp.array([jnp.pi/4, jnp.pi/4, jnp.pi/2])
    rotated_coords = rotate_rpy(coords, rpy)

    # Project
    projection = project(model, rotated_coords, shape)
    image = interpolate(projection, method="linear", fill_value=0.0)
    print((image - projection).max(), (image - projection).min())
    #print(jnp.unique(image, return_counts=True))

    # Normalize
    #image = normalize(image)
    #image = image.at[jnp.where(image != 0)].set(jnp.nan)

    fig, axes = plt.subplots(ncols=3)
    ax1, ax2, ax3 = axes
    ax1.set(title="Projection")
    ax2.set(title="Interpolated projection")
    ax3.set(title="Difference")
    im1 = ax1.imshow(projection, cmap="gray")
    ax2.imshow(image, cmap="gray")
    ax3.imshow(image - projection, cmap="gray")
    # fig.colorbar(im1, ax=ax1)
    plt.show()
