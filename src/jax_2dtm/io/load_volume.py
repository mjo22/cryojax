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
    Read 3D template to jax array.

    Parameters
    ----------
    filename :
        Path to 3D template.

    Returns
    -------
    template : shape `(N1, N2, N3)`.
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
    filename :
        Path to EMDB .map file

    Returns
    -------
    volume : shape `(N1, N2, N3)`
        3D model in cartesian coordinates
    """
    ccp4_map = gemmi.read_ccp4_map(filename)
    volume = jnp.array(ccp4_map.grid, copy=False)

    return volume
