#!/usr/bin/env python3
"""
Routines for reading 3D models to numpy arrays
"""

import os
import mrcfile
import jax.numpy as jnp


def read_mrc(filename):
    """
    Read 3D template to jax numpy array.

    Parameters
    ----------
    filename : `str`
        Path to 3D template.
    Returns
    -------
    template : `jnp.ndarray`, shape `(N1, N2, N3)`
        3D model in cartesian coordinates.
    """
    with mrcfile.open(filename) as mrc:
        template = jnp.array(mrc.data)

    return template


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    template = read_mrc("./example/6dpu_14pf_bfm1_ps1_1.mrc")

    fig, ax = plt.subplots(ncols=3)
    titles = ['x projection', 'y projection', 'z projection']
    for idx in range(len(titles)):
        title = titles[idx]
        projection = jnp.sum(template[:, :, :], axis=idx)
        ax[idx].imshow(projection, cmap='gray')
        ax[idx].set_title(title)

    fig.suptitle('14 protofilament microtubule template')
    plt.show()
