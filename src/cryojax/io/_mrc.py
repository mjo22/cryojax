"""
Routines for working with MRC files.
"""

import mrcfile
import numpy as np
from jaxtyping import Float, Array


def read_array_with_spacing_from_mrc(
    filename: str,
    mmap: bool = False,
) -> tuple[Float[np.ndarray, "..."], float]:
    """Read MRC data to a numpy array, including the grid spacing
    (the voxel size or pixel size).

    !!! note
        This function only supports grid spacing that is the same
        in all dimensions.

    **Arguments:**

    'filename': Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'array': The array stored in the Mrcfile.

    'grid_spacing': The voxel size or pixel size of `data`.
    """
    array, grid_spacing = _read_array_from_mrc(filename, get_spacing=True, mmap=mmap)

    return array.astype(np.float64), float(grid_spacing)


def read_array_from_mrc(filename: str, mmap: bool = False) -> Float[np.ndarray, "..."]:
    """Read MRC data to a numpy array.

    **Arguments:**

    'filename' : Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'array' : The array stored in the Mrcfile.
    """
    array = _read_array_from_mrc(filename, get_spacing=False, mmap=mmap)

    return array.astype(np.float64)


def write_image_stack_to_mrc(
    image_stack: Float[Array, "M N1 N2"],
    pixel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str,
):
    """Write an image stack to an MRC file.

    **Arguments:**

    `image_stack`: The image stack, where the leading dimension indexes the image.

    `pixel_size`: The pixel size of the images in `image_stack`.

    `filename`: The output filename.
    """
    # Convert image stack and pixel size to numpy arrays.
    image_stack = np.asarray(image_stack)
    pixel_size = np.asarray(pixel_size)
    # Convert image stack to MRC xyz conventions
    image_stack = image_stack.transpose(0, 2, 1)
    # Create new file and write
    with mrcfile.open(filename, mode="w+") as mrc:
        mrc.set_data(image_stack)
        mrc.set_image_stack()
        mrc.voxel_size = (1.0, pixel_size, pixel_size)


def write_voxel_grid_to_mrc(
    voxel_grid: Float[Array, "N1 N2 N3"],
    voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str,
):
    """Write a voxel grid to an MRC file.

    **Arguments:**

    `voxel_grid`: The voxel grid as a JAX array.

    `voxel_size`: The voxel size of the `voxel_grid`.

    `filename`: The output filename.
    """
    # Convert volume and voxel size to numpy arrays.
    voxel_grid = np.asarray(voxel_grid)
    voxel_size = np.asarray(voxel_size)
    # Convert volume to MRC xyz conventions
    voxel_grid = voxel_grid.transpose(2, 1, 0)
    # Create new file and write
    with mrcfile.open(filename, mode="w+") as mrc:
        mrc.set_data(voxel_grid)
        mrc.set_volume()
        mrc.voxel_size = (voxel_size, voxel_size, voxel_size)


def _read_array_from_mrc(
    filename: str, get_spacing: bool, mmap: bool
) -> Float[np.ndarray, "..."] | tuple[Float[np.ndarray, "..."], float]:
    # Read MRC
    open = mrcfile.mmap if mmap else mrcfile.open
    with open(filename, mode="r") as mrc:
        array = mrc.data
        # Convert to cryojax xyz conventions
        if mrc.is_single_image():
            array = array.transpose(1, 0)
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.y, mrc.voxel_size.x], dtype=float
            )
        elif mrc.is_image_stack():
            array = array.transpose(0, 2, 1)
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.y, mrc.voxel_size.x], dtype=float
            )
        elif mrc.is_volume():
            array = array.transpose(2, 1, 0)
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x],
                dtype=float,
            )
        elif mrc.is_volume_stack():
            array = array.transpose(0, 3, 2, 1)
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x],
                dtype=float,
            )
        else:
            raise ValueError(
                f"Mrcfile could not be identified as an image, image stack, volume, or volume stack. "
                "Run mrcfile.validate(...) to make sure {filename} is a valid MRC file."
            )

        if get_spacing:
            # Only allow the same spacing in each direction
            assert all(
                grid_spacing_per_dimension != np.zeros(grid_spacing_per_dimension.shape)
            ), "Mrcfile.voxel_size must be set to use cryojax.io.read_array_with_spacing_from_mrc. Try running cryojax.io.read_array_from_mrc instead."
            assert all(
                grid_spacing_per_dimension == grid_spacing_per_dimension[0]
            ), "Mrcfile.voxel_size must be same in all dimensions."
            grid_spacing = grid_spacing_per_dimension[0]

            return array, grid_spacing
        else:
            # ... otherwise, just return
            return array
