"""
Routines for reading 3D models into arrays.
"""

__all__ = ["load_mrc", "load_grid_as_cloud"]

import mrcfile
import numpy as np
import jax.numpy as jnp
from ..simulator import coordinatize, Cloud, ImageConfig
from ..types import ArrayLike


def load_grid_as_cloud(filename: str, config: ImageConfig, **kwargs) -> Cloud:
    """
    Read a 3D template on a cartesian grid
    to a ``Cloud``.

    Parameters
    ----------
    filename :
        Path to template.
    config :
        Image configuration.
    kwargs :
        Keyword arguments passed to
        ``jax_2dtm.simulator.coordinatize``.

    Returns
    -------
    cloud :
        Point cloud generated from the 3D template.
    """
    template = load_mrc(filename)
    depth = template.shape[2]
    assert all(Ni == depth for Ni in template.shape)
    box_size = (
        jnp.array((*config.shape, depth), dtype=float) * config.pixel_size
    )
    cloud = Cloud(
        *coordinatize(template, config.pixel_size, **kwargs), box_size
    )

    return cloud


def load_mrc(filename: str) -> ArrayLike:
    """
    Read template to ``numpy`` array.

    Parameters
    ----------
    filename :
        Path to template.

    Returns
    -------
    template :
        Model in cartesian coordinates.
    """
    with mrcfile.open(filename) as mrc:
        template = np.array(mrc.data)

    return template
