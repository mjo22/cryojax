# -*- coding: utf-8 -*-
"""
Routines for reading 3D models to numpy arrays
"""

__all__ = ["load_mrc", "load_ccp4"]

import mrcfile
import gemmi
import jax.numpy as jnp
from jax_2dtm.types import Array


def load_mrc(filename: str) -> Array:
    """
    Read 3D template or 2D image to jax
    numpy array.

    Parameters
    ----------
    filename : `str`
        Path to 3D template.
    Returns
    -------
    template : `jnp.ndarray`, shape `(N1, N2, N3)` or `(N2, N3)`.
        3D model in cartesian coordinates.
    """
    with mrcfile.open(filename) as mrc:
        template = jnp.array(mrc.data)

    return template


def load_ccp4(filename : str) -> Array:
    """
    Read 3D EMDB model into numpy array

    Parameters
    ----------
    filename : `str`
        Path to EMDB .map file

    Returns
    -------
    volume : `jnp.ndarray`, shape `(N1, N2, N3)`
        3D model in cartesian coordinates
    """
    ccp4_map = gemmi.read_ccp4_map(filename)
    volume = jnp.array(ccp4_map.grid, copy=False)

    return volume


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    template = load_mrc("./example/6dpu_14pf_bfm1_ps1_1.mrc")

    fig, ax = plt.subplots(ncols=3)
    titles = ['x projection', 'y projection', 'z projection']
    for idx in range(len(titles)):
        title = titles[idx]
        projection = jnp.sum(template[:, :, :], axis=idx)
        ax[idx].imshow(projection, cmap='gray')
        ax[idx].set_title(title)

    fig.suptitle('14 protofilament microtubule template')
    plt.show()
