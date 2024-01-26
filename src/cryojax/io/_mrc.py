"""
Routines for working with MRC files.
"""

__all__ = ["load_mrc"]

import numpy as np
import mrcfile


def load_mrc(filename: str) -> tuple[np.ndarray, float]:
    """
    Read MRC data to ``numpy`` array.

    Parameters
    ----------
    filename : `str`
        Path to data.

    Returns
    -------
    data : `ArrayLike`, shape `(N1, N2, N3)` or `(N1, N2)`
        Model in cartesian coordinates.
    voxel_size : `ArrayLike`, shape `(3,)` or `(2,)`
        The voxel_size in each dimension, stored
        in the MRC file.
    """
    # Read MRC
    with mrcfile.open(filename) as mrc:
        data = np.asarray(mrc.data, dtype=float)
        if data.ndim == 2:
            # Change how template sits in box to match cisTEM
            data = np.transpose(data, axes=[1, 0])
            voxel_size = np.asarray(
                [mrc.voxel_size.y, mrc.voxel_size.x], dtype=float
            )
        elif data.ndim == 3:
            # Change how template sits in box to match cisTEM
            data = np.transpose(data, axes=[1, 2, 0])
            voxel_size = np.asarray(
                [mrc.voxel_size.y, mrc.voxel_size.z, mrc.voxel_size.x],
                dtype=float,
            )
        else:
            raise NotImplementedError(
                "MRC files with 2D and 3D data are supported."
            )

    assert all(
        voxel_size != np.zeros(data.ndim)
    ), "MRC file must set the voxel size."
    assert all(
        voxel_size == voxel_size[0]
    ), "Voxel size must be same in all dimensions."

    return data, voxel_size[0]
