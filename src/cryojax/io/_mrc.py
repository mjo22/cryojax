"""
Routines for working with MRC files.
"""

import numpy as np
import mrcfile


def read_image_or_volume_with_spacing_from_mrc(
    filename: str,
) -> tuple[np.ndarray, float]:
    """
    Read MRC data to ``numpy`` array.

    Parameters
    ----------
    filename : `str`
        Path to data.

    Returns
    -------
    data : `np.ndarray`, shape `(N1, N2, N3)` or `(N1, N2)`
        Image or volume array.
    grid_spacing : `float`, shape `(3,)` or `(2,)`
        The voxel or pixel size stored in the MRC file.
    """
    # Read MRC
    with mrcfile.open(filename) as mrc:
        data = np.asarray(mrc.data, dtype=float)
        if data.ndim == 2:
            grid_spacing = np.asarray([mrc.voxel_size.y, mrc.voxel_size.x], dtype=float)
        elif data.ndim == 3:
            # Change how grid sits in box to match cisTEM
            data = np.transpose(data, axes=[2, 1, 0])
            grid_spacing = np.asarray(
                [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x],
                dtype=float,
            )
        else:
            raise NotImplementedError("MRC files with 2D and 3D data are supported.")

    assert all(grid_spacing != np.zeros(data.ndim)), "MRC file must set the voxel size."
    assert all(
        grid_spacing == grid_spacing[0]
    ), "Voxel size must be same in all dimensions."

    return data, grid_spacing[0]


def read_image_or_volume_from_mrc(filename: str) -> np.ndarray:
    """
    Read MRC data to ``numpy`` array.

    Parameters
    ----------
    filename : `str`
        Path to data.

    Returns
    -------
    data : `np.ndarray`, shape `(N1, N2, N3)` or `(N1, N2)`
         Image or volume array.
    """
    # Read MRC
    with mrcfile.open(filename) as mrc:
        data = np.asarray(mrc.data, dtype=float)
        if data.ndim == 3:
            # Change how grid sits in box to match cisTEM
            data = np.transpose(data, axes=[2, 1, 0])

    return data
