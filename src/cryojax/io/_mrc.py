"""
Routines for working with MRC files.
"""

import mrcfile
import pathlib
import numpy as np
from jaxtyping import Float, Array
from typing import Optional
from pathlib import Path


def read_image_with_pixel_size_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> tuple[Float[np.ndarray, "..."], float]:
    """Read an MRC image to a numpy array, including the pixel size.

    !!! note
        This function only supports a pixel size that is the same
        in all dimensions.

    **Arguments:**

    'filename': Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'image': The image stored in the Mrcfile.

    'pixel_size': The pixel size of `image`.
    """
    image, pixel_size = _read_array_from_mrc(
        filename,
        expected_data_format="image",
        get_spacing=True,
        mmap=mmap,
        permissive=permissive,
    )

    return image, pixel_size


def read_image_stack_with_pixel_size_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> tuple[Float[np.ndarray, "..."], float]:
    """Read an MRC image stack to a numpy array, including the pixel size.

    !!! note
        This function only supports a pixel size that is the same
        in all dimensions.

    **Arguments:**

    'filename': Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'image_stack': The image stored in the Mrcfile.

    'pixel_size': The pixel size of `image_stack`.
    """
    image_stack, pixel_size = _read_array_from_mrc(
        filename,
        expected_data_format="image_stack",
        get_spacing=True,
        mmap=mmap,
        permissive=permissive,
    )

    return image_stack, pixel_size


def read_volume_with_voxel_size_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> tuple[Float[np.ndarray, "..."], float]:
    """Read an MRC volume to a numpy array, including the voxel size.

    !!! note
        This function only supports a voxel size that is the same
        in all dimensions.

    **Arguments:**

    'filename': Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'volume': The volume stored in the Mrcfile.

    'voxel_size': The voxel size of `volume`.
    """
    volume, voxel_size = _read_array_from_mrc(
        filename,
        expected_data_format="volume",
        get_spacing=True,
        mmap=mmap,
        permissive=permissive,
    )

    return volume, voxel_size


def read_volume_stack_with_voxel_size_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> tuple[Float[np.ndarray, "..."], float]:
    """Read an MRC volume stack to a numpy array, including the voxel size.

    !!! note
        This function only supports a voxel size that is the same
        in all dimensions.

    **Arguments:**

    'filename': Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'volume_stack': The volume stack stored in the Mrcfile.

    'voxel_size': The voxel size of `volume`.
    """
    volume_stack, voxel_size = _read_array_from_mrc(
        filename,
        expected_data_format="volume_stack",
        get_spacing=True,
        mmap=mmap,
        permissive=permissive,
    )

    return volume_stack, voxel_size


def read_image_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> Float[np.ndarray, "..."]:
    """Read an MRC single image to a numpy array.

    **Arguments:**

    'filename' : Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'image' : The image stored in the Mrcfile.
    """
    image = _read_array_from_mrc(
        filename,
        expected_data_format="image",
        get_spacing=False,
        mmap=mmap,
        permissive=permissive,
    )

    return image


def read_image_stack_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> Float[np.ndarray, "..."]:
    """Read an MRC image stack to a numpy array.

    **Arguments:**

    'filename' : Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'image_stack' : The image stack stored in the Mrcfile.
    """
    image_stack = _read_array_from_mrc(
        filename,
        expected_data_format="image_stack",
        get_spacing=False,
        mmap=mmap,
        permissive=permissive,
    )

    return image_stack


def read_volume_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> Float[np.ndarray, "..."]:
    """Read an MRC volume to a numpy array.

    **Arguments:**

    'filename' : Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'volume' : The volume stored in the Mrcfile.
    """
    volume = _read_array_from_mrc(
        filename,
        expected_data_format="volume",
        get_spacing=False,
        mmap=mmap,
        permissive=permissive,
    )

    return volume


def read_volume_stack_from_mrc(
    filename: str | Path,
    *,
    mmap: bool = False,
    permissive: bool = False,
) -> Float[np.ndarray, "..."]:
    """Read an MRC volume stack to a numpy array.

    **Arguments:**

    'filename' : Path to data.

    `mmap`: Whether or not to open the data as a `numpy.memmap` array.

    **Returns:**

    'volume_stack' : The volume stack stored in the Mrcfile.
    """
    volume_stack = _read_array_from_mrc(
        filename,
        expected_data_format="volume_stack",
        get_spacing=False,
        mmap=mmap,
        permissive=permissive,
    )

    return volume_stack


def write_image_to_mrc(
    image: Float[Array, "N1 N2"],
    pixel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str | Path,
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    """Write an image stack to an MRC file.

    **Arguments:**

    `image`: The image stack, where the leading dimension indexes the image.

    `pixel_size`: The pixel size of the images in `image_stack`.

    `filename`: The output filename.

    `overwrite`: If `True`, overwrite an existing file.

    `compression`: See `mrcfile.new` for documentation.
    """
    # Validate filename as MRC path and get suffix
    suffix = _validate_filename_and_return_suffix(filename)
    if suffix != ".mrc":
        raise IOError(
            "The suffix for an image in MRC format must be .mrc. "
            f"Instead, got {suffix}."
        )
    if image.ndim != 2:
        raise ValueError("image.ndim must be equal to 2.")
    # Convert image stack and pixel size to numpy arrays.
    image = np.asarray(image)
    pixel_size = np.asarray(pixel_size)
    # Create new file and write
    with mrcfile.new(filename, compression=compression, overwrite=overwrite) as mrc:
        mrc.set_data(image)
        mrc.voxel_size = pixel_size


def write_image_stack_to_mrc(
    image_stack: Float[Array, "M N1 N2"],
    pixel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str | Path,
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    """Write an image stack to an MRC file.

    **Arguments:**

    `image_stack`: The image stack, where the leading dimension indexes the image.

    `pixel_size`: The pixel size of the images in `image_stack`.

    `filename`: The output filename.

    `overwrite`: If `True`, overwrite an existing file.

    `compression`: See `mrcfile.new` for documentation.
    """
    # Validate filename as MRC path and get suffix
    suffix = _validate_filename_and_return_suffix(filename)
    if suffix != ".mrcs":
        raise IOError(
            "The suffix for an image stack in MRC format must be .mrcs. "
            f"Instead, got {suffix}."
        )
    if image_stack.ndim != 3:
        raise ValueError("image_stack.ndim must be equal to 3.")
    # Convert image stack and pixel size to numpy arrays.
    image_stack = np.asarray(image_stack)
    pixel_size = np.asarray(pixel_size)
    # Create new file and write
    with mrcfile.new(filename, compression=compression, overwrite=overwrite) as mrc:
        mrc.set_data(image_stack)
        mrc.set_image_stack()
        mrc.voxel_size = (1.0, pixel_size, pixel_size)


def write_volume_to_mrc(
    voxel_grid: Float[Array, "N1 N2 N3"],
    voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str | Path,
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    """Write a voxel grid to an MRC file.

    **Arguments:**

    `voxel_grid`: The voxel grid as a JAX array.

    `voxel_size`: The voxel size of the `voxel_grid`.

    `filename`: The output filename.

    `overwrite`: If `True`, overwrite an existing file.

    `compression`: See `mrcfile.new` for documentation.
    """
    # Validate filename as MRC path and get suffix
    suffix = _validate_filename_and_return_suffix(filename)
    if suffix != ".mrc":
        raise IOError(
            "The suffix for a volume in MRC format must be .mrc. "
            f"Instead, got {suffix}."
        )
    if voxel_grid.ndim != 3:
        raise ValueError("voxel_grid.ndim must be equal to 3.")
    # Convert volume and voxel size to numpy arrays.
    voxel_grid = np.asarray(voxel_grid)
    voxel_size = np.asarray(voxel_size)
    # Convert volume to MRC xyz conventions
    voxel_grid = voxel_grid.transpose(2, 1, 0)
    # Create new file and write
    with mrcfile.new(filename, compression=compression, overwrite=overwrite) as mrc:
        mrc.set_data(voxel_grid)
        mrc.set_volume()
        mrc.voxel_size = (voxel_size, voxel_size, voxel_size)


def _read_array_from_mrc(
    filename: str | Path,
    expected_data_format: str,
    get_spacing: bool,
    mmap: bool,
    permissive: bool,
) -> Float[np.ndarray, "..."] | tuple[Float[np.ndarray, "..."], float]:
    # Validate filename as MRC path and get suffix
    suffix = _validate_filename_and_return_suffix(filename)
    # Read MRC
    open = mrcfile.mmap if mmap else mrcfile.open
    with open(filename, mode="r", permissive=permissive) as mrc:
        array = mrc.data
        # Convert to cryojax xyz conventions
        if expected_data_format == "image":
            if suffix != ".mrc":
                raise IOError(
                    "The suffix for a single image in MRC format must be .mrc. "
                    f"Instead, got {suffix}."
                )
            if array.ndim != 2:
                raise IOError(
                    "Tried to read a single image in MRC format, which should have ndim = 2."
                    f"Instead, the array had ndim = {array.ndim}"
                )
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.y, mrc.voxel_size.x], dtype=float
            )
        elif expected_data_format == "image_stack":
            if suffix != ".mrcs":
                raise IOError(
                    "The suffix for an image stack in MRC format must be .mrcs. "
                    f"Instead, got {suffix}."
                )
            if array.ndim != 3:
                raise IOError(
                    "Tried to read an image stack in MRC format, which should have ndim = 3."
                    f"Instead, the array had ndim = {array.ndim}"
                )
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.y, mrc.voxel_size.x], dtype=float
            )
        elif expected_data_format == "volume":
            if suffix != ".mrc":
                raise IOError(
                    "The suffix for a volume in MRC format must be .mrc. "
                    f"Instead, got {suffix}."
                )
            if array.ndim != 3:
                raise IOError(
                    "Tried to read a volume in MRC format, which should have ndim = 3."
                    f"Instead, the array had ndim = {array.ndim}"
                )
            array = array.transpose(2, 1, 0)
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x],
                dtype=float,
            )
        elif expected_data_format == "volume_stack":
            if suffix != ".mrcs":
                raise IOError(
                    "The suffix for a volume stack in MRC format must be .mrcs. "
                    f"Instead, got {suffix}."
                )
            if array.ndim != 4:
                raise IOError(
                    "Tried to read a volume stack in MRC format, which should have ndim = 4."
                    f"Instead, the array had ndim = {array.ndim}"
                )
            array = array.transpose(0, 3, 2, 1)
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x],
                dtype=float,
            )
        else:
            raise Exception(
                "Could not read array from MRC file. This is a source code issue, so please alert "
                "developers on github."
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

            return array.astype(np.float64), float(grid_spacing)
        else:
            # ... otherwise, just return
            return array.astype(np.float64)


def _validate_filename_and_return_suffix(filename: str | Path):
    # Get suffixes
    suffixes = pathlib.Path(filename).suffixes
    # Make sure that leading suffix is valid MRC suffix
    if len(suffixes) == 0 or suffixes[0] not in [".mrc", ".mrcs"]:
        raise IOError(
            f"Filename should include .mrc or .mrcs suffix. Got filename {filename}."
        )
    return suffixes[0]
