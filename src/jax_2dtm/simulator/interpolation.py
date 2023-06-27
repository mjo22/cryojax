#!/usr/bin/env python3
"""
Routines to interpolate an irregular grid onto a rectangular grid.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.interpolate import griddata
from coordinates import coordinatize


def interpolate(projection, **kwargs):
    """
    Interpolate rotated image.

    Arguments
    ---------
    image : jnp.ndarray, shape `(N2, N3)`
        Template projection with discontinuities to be
        interpolated onto regular grid.

    Returns
    -------
    interpolated : jnp.ndarray, shape `(N2, N3)`
        Interpolated projection.
    """
    # Coordinates
    shape = projection.shape
    flat, coords = coordinatize(projection)
    flat, coords = np.asarray(flat), np.asarray(coords)
    # Convert uninterpolated image to point cloud
    mask = flat != 0.
    values, points = flat[mask], coords[mask]
    # Interpolate
    result = griddata(points, values, coords, **kwargs)
    image = result.reshape(shape)

    return jnp.asarray(image)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from jax import random

    shape = (100, 100)
    N = shape[0] * shape[1]
    key = random.PRNGKey(0)
    xv, yv = jnp.meshgrid(*[np.arange(Ni)-Ni//2 for Ni in shape], indexing="ij")
    rv = jnp.sqrt(xv**2 + yv**2).ravel()
    idxs = random.choice(key, jnp.arange(N), shape=(N//10,))
    image = rv.at[idxs].set(0.).reshape(shape)

    # Interpolate
    interpolated = interpolate(image, method="cubic")

    fig, axes = plt.subplots(ncols=3)
    images = [rv.reshape(shape), image, interpolated]
    titles = ["Original image", "Set pixels to zero", "Interpolated image"]
    for idx, ax in enumerate(axes):
        ax.set(title=titles[idx])
        im = ax.imshow(images[idx], vmin=0, vmax=70, cmap="gray")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)
    fig.colorbar(im, cax=cax)
    plt.show()
