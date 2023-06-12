#!/usr/bin/env python3
#
# Cross-correlation of a template projection and micrograph


import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
from jax import jit


@jit
def cross_correlate(micrograph, projection):
    """
    Compute cross correlation of micrograph with template
    projection.
    """
    cc = fftconvolve(projection, micrograph, mode="valid")

    return cc


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from template import read_mrc
    from coordinates import coordinatize
    from rotations import rotate_rpy
    from interpolation import interpolate
    from imaging import project

    micrograph = read_mrc("./example/MO_Car4_gr1_0063.mrc")
    template = read_mrc("./example/6dpu_14pf_bfm1_ps1_1.mrc")

    shape = template.shape[1:]
    # Read coordinates
    model, coords = coordinatize(template)
    # Apply rotation
    rpy = jnp.array([0, 0, 0])
    rotated_coords = rotate_rpy(coords, rpy)
    # Project
    image = project(model, rotated_coords, shape)
    image = interpolate(image)

    cc = cross_correlate(micrograph, image)

    fig, ax = plt.subplots()
    im = ax.imshow(cc, cmap="viridis", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.show()
