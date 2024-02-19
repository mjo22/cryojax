"""
Routines for working with MRC files.
"""

import mrcfile
import numpy as np
from jaxtyping import Float


def read_array_with_spacing_from_mrc(
    filename: str,
    mmap: bool = False,
) -> tuple[Float[np.ndarray, "..."], Float[np.ndarray, ""]]:
    """Read MRC data to a numpy array, including the grid spacing
    (the voxel size or pixel size).

    !!! note
        This function only supports grid spacing that is the same
        in all dimensions.

    **Arguments:**

    'filename': Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'data': The array stored in the Mrcfile.

    'grid_spacing': The voxel size or pixel size of `data`.
    """
    # Read MRC
    open = mrcfile.mmap if mmap else mrcfile.open
    with open(filename, mode="r") as mrc:
        data = mrc.data
        if mrc.is_single_image() or mrc.is_image_stack():
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y], dtype=float
            )
        elif mrc.is_volume() or mrc.is_volume_stack():
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
                dtype=float,
            )
        else:
            raise ValueError(
                f"Mrcfile could not be identified as an image, image stack, volume, or volume stack. Run mrcfile.validate(...) to make sure {filename} is a valid MRC file."
            )

    assert all(
        grid_spacing_per_dimension != np.zeros(grid_spacing_per_dimension.shape)
    ), "Mrcfile.voxel_size must be set to use cryojax.io.read_array_with_spacing_from_mrc. Try running cryojax.io.read_array_from_mrc instead."
    assert all(
        grid_spacing_per_dimension == grid_spacing_per_dimension[0]
    ), "Mrcfile.voxel_size must be same in all dimensions."
    # Set the grid spacing
    grid_spacing = grid_spacing_per_dimension[0]

    return data.astype(np.float64), grid_spacing


def read_array_from_mrc(filename: str, mmap: bool = False) -> Float[np.ndarray, "..."]:
    """Read MRC data to a numpy array.

    **Arguments:**

    'filename' : Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'data' : The array stored in the Mrcfile.
    """
    # Read MRC
    open = mrcfile.mmap if mmap else mrcfile.open
    with open(filename, mode="r") as mrc:
        data = mrc.data

    return data.astype(np.float64)
