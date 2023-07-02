"""
Routines for reading 3D models to numpy arrays
"""

__all__ = ["load_mrc"]

import mrcfile
import numpy as np
from jax_2dtm.types import ArrayLike


def load_mrc(filename: str) -> ArrayLike:
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
        template = np.array(mrc.data)

    return template
