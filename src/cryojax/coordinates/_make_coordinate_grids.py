"""
Functions for creating coordinate systems.
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def make_coordinate_grid(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
) -> Float[Array, "*shape ndim"]:
    """
    Create a real-space cartesian coordinate system on a grid.

    **Arguments:**

    - `shape`:
        Shape of the grid, with `ndim = len(shape)`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.

    **Returns:**

    A cartesian coordinate system in real space.
    """
    coordinate_grid = _make_coordinates_or_frequencies(
        shape, grid_spacing=grid_spacing, real_space=True
    )
    return coordinate_grid


def make_frequency_grid(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    get_rfftfreqs: bool = True,
) -> Float[Array, "*shape ndim"]:
    """Create a fourier-space cartesian coordinate system on a grid.
    The zero-frequency component is in the corner.

    **Arguments:**

    - `shape`:
        Shape of the grid, with `ndim = len(shape)`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.
    - `get_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    A cartesian coordinate system in frequency space.
    """
    frequency_grid = _make_coordinates_or_frequencies(
        shape,
        grid_spacing=grid_spacing,
        real_space=False,
        get_rfftfreqs=get_rfftfreqs,
    )
    return frequency_grid


def make_frequency_slice(
    shape: tuple[int, int],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    get_rfftfreqs: bool = True,
) -> Float[Array, "1 {shape[0]} {shape[1]} 3"]:
    """Create a fourier-space cartesian coordinate system on a grid, where
    zero-frequency component is in the *center* of the grid.

    !!! warning
        In the function `make_frequency_grid`, the convention is that
        the grid is returned with the zero frequency component is in the
        corner. In this function, as mentioned above, frequency slices are
        returned with the zero frequency component in the center. To convert
        between the two conventions, run

        ```python
        import jax.numpy as jnp
        from cryojax.coordinates import make_frequency_slice

        frequency_slice_with_zero_in_center = make_frequency_slice((100, 100)) # Shape (1, 100, 100, 3)
        frequency_slice_with_zero_in_corner = jnp.fft.ifftshift(frequency_slice_with_zero_in_center, axes=(1, 2))
        ```

        The reason for the difference is so that this function can be used to
        directly pass a `frequency_slice` to the `cryojax.simulator.FourierVoxelGridPotential`,
        which requires that the zero is in the center of the grid.

    **Arguments:**

    - `shape`:
        Shape of the frequency slice, e.g. `shape = (100, 100)`.
    - `grid_spacing`:
        The grid spacing (i.e. voxel size), in units of length.
    - `get_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    The central, $q_z = 0$ slice of a 3D frequency grid $(q_x, q_y, q_z)$, where
    zero-frequency component is in the *center* of the grid.
    """  # noqa: E501
    frequency_slice = make_frequency_grid(
        shape, grid_spacing, get_rfftfreqs=get_rfftfreqs
    )
    if get_rfftfreqs:
        frequency_slice = jnp.fft.fftshift(frequency_slice, axes=(0,))
    else:
        frequency_slice = jnp.fft.fftshift(frequency_slice, axes=(0, 1))
    frequency_slice = jnp.expand_dims(
        jnp.pad(
            frequency_slice,
            ((0, 0), (0, 0), (0, 1)),
            mode="constant",
            constant_values=0.0,
        ),
        axis=0,
    )
    return frequency_slice


def make_1d_coordinate_grid(
    size: int,
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
) -> Float[Array, "*shape ndim"]:
    """
    Create a 1D real-space cartesian coordinate array.

    **Arguments:**

    - `size`:
        Size of the coordinate array.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.

    **Returns:**

    A 1D cartesian coordinate array in real space.
    """
    coordinate_array = _make_coordinates_or_frequencies_1d(
        size, grid_spacing=grid_spacing, real_space=True
    )
    return coordinate_array


def make_1d_frequency_grid(
    size: int,
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    get_rfftfreqs: bool = True,
) -> Float[Array, "*shape ndim"]:
    """Create a 1D fourier-space cartesian coordinate array.
    If `get_rfftfreqs = False`, the zero-frequency component is in the beginning.

    Arguments
    ---------
    - `size`:
        Size of the coordinate array.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.
    - `get_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    A 1D cartesian coordinate array in frequency space.
    """
    frequency_array = _make_coordinates_or_frequencies_1d(
        size,
        grid_spacing=grid_spacing,
        real_space=False,
        rfftfreq=get_rfftfreqs,
    )
    return frequency_array


def _make_coordinates_or_frequencies(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    real_space: bool = False,
    get_rfftfreqs: bool = True,
) -> Float[Array, "*shape ndim"]:
    ndim = len(shape)
    coords1D = []
    for idx in range(ndim):
        if real_space:
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, real_space
            )
        else:
            if not get_rfftfreqs:
                rfftfreq = False
            else:
                rfftfreq = False if idx < ndim - 1 else True
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, real_space, rfftfreq
            )
        coords1D.append(c1D)
    if ndim == 2:
        y, x = coords1D
        xv, yv = jnp.meshgrid(x, y, indexing="xy")
        coords = jnp.stack([xv, yv], axis=-1)
    elif ndim == 3:
        z, y, x = coords1D
        xv, yv, zv = jnp.meshgrid(x, y, z, indexing="xy")
        xv, yv, zv = [
            jnp.transpose(rv, axes=[2, 0, 1]) for rv in [xv, yv, zv]
        ]  # Change axis ordering to [z, y, x]
        coords = jnp.stack([xv, yv, zv], axis=-1)
    else:
        raise ValueError(
            "Only 2D and 3D coordinate grids are supported. "
            f"Tried to create a grid of shape {shape}."
        )

    return coords


def _make_coordinates_or_frequencies_1d(
    size: int,
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""],
    real_space: bool = False,
    rfftfreq: Optional[bool] = None,
) -> Float[Array, " size"]:
    """One-dimensional coordinates in real or fourier space"""
    if real_space:
        make_1d = lambda size, dx: jnp.fft.fftshift(jnp.fft.fftfreq(size, 1 / dx)) * size
    else:
        if rfftfreq is None:
            raise ValueError("Argument rfftfreq cannot be None if real_space=False.")
        else:
            fn = jnp.fft.rfftfreq if rfftfreq else jnp.fft.fftfreq
            make_1d = lambda size, dx: fn(size, grid_spacing)

    return make_1d(size, grid_spacing)
