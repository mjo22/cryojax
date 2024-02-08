"""
Routines for working with MRC files.
"""

import mrcfile
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from ..typing import Real_


def read_array_with_spacing_from_mrc(
    filename: str,
) -> tuple[Array, Real_]:
    """Read MRC data to a JAX array, including the grid spacing
    (the voxel or pixel size).

    !!! note
        This function does not support grid spacing that is not
        the same in all dimensions

    **Arguments:**

    'filename' : Path to data.

    **Returns:**

    'data' : The array stored in the MRC file.

    'grid_spacing' : The voxel size or pixel size of `data`.
    """
    # Read MRC
    with mrcfile.open(filename) as mrc:
        data = np.asarray(mrc.data, dtype=float)
        if data.ndim == 2:
            grid_spacing = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y], dtype=float)
        elif data.ndim == 3:
            grid_spacing = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
                dtype=float,
            )
        else:
            raise NotImplementedError("MRC files with 2D and 3D data are supported.")

    assert all(grid_spacing != np.zeros(data.ndim)), "MRC file must set the voxel size."
    assert all(
        grid_spacing == grid_spacing[0]
    ), "Grid spacing must be same in all dimensions."

    return jnp.asarray(data), jnp.asarray(grid_spacing[0])


def read_array_from_mrc(filename: str) -> Array:
    """Read MRC data to a JAX array.

    **Arguments:**

    'filename' : Path to data.

    **Returns:**

    'data' : The array stored in the MRC file.
    """
    # Read MRC
    with mrcfile.open(filename) as mrc:
        data = np.asarray(mrc.data, dtype=float)

    return jnp.asarray(data)
