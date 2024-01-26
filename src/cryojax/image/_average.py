"""
Routines to compute radial averages of images.
"""

__all__ = ["radial_average"]

from typing import overload
from functools import partial

import jax
import jax.numpy as jnp

from ..typing import RealVector, Vector, Image, RealImage, Volume, RealVolume


@overload
def radial_average(
    image: Image,
    radial_grid: RealImage,
    bins: RealVector,
) -> Vector: ...


@overload
def radial_average(
    image: Volume,
    radial_grid: RealVolume,
    bins: RealVector,
) -> Vector: ...


@overload
def radial_average(
    image: Image,
    radial_grid: RealImage,
    bins: RealVector,
    *,
    to_grid: bool,
    interpolation_mode: str,
) -> tuple[Vector, Image]: ...


@overload
def radial_average(
    image: Volume,
    radial_grid: RealVolume,
    bins: RealVector,
    *,
    to_grid: bool,
    interpolation_mode: str,
) -> tuple[Vector, Volume]: ...


@partial(jax.jit, static_argnames=["to_grid", "interpolation_mode"])
def radial_average(
    image: Image | Volume,
    radial_grid: RealImage | RealVolume,
    bins: RealVector,
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
) -> Vector | tuple[Vector, Image | Volume]:
    """
    Radially average vectors r with a given magnitude
    coordinate system |r|.

    Arguments
    ---------
    image :
        Two-dimensional image.
    radial_grid :
        Radial coordinate system of image.
    bins :
        Radial bins for averaging. These
        must be evenly spaced.
    to_grid :
        If ``False``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on the
        grid.
    interpolation_mode :
        If ``"linear"``, evaluate on the grid using linear
        interpolation. If ``False``,

    Returns
    -------
    average :
        Radial average of image.
    """
    bins = jnp.asarray(bins)
    # Discretize the radial grid
    digitized_radial_grid = jnp.digitize(radial_grid, bins, right=True)
    # Compute the radial profile as the average value of the image in each bin
    average_as_profile = jnp.bincount(
        digitized_radial_grid.ravel(),
        weights=image.ravel(),
        length=bins.size,
    ) / jnp.bincount(digitized_radial_grid.ravel(), length=bins.size)
    # Interpolate to a grid or return the profile
    if to_grid:
        if interpolation_mode == "nearest":
            average_as_grid = jnp.take(
                average_as_profile,
                digitized_radial_grid,
                mode="clip",
            )
        elif interpolation_mode == "linear":
            average_as_grid = jnp.interp(
                radial_grid.ravel(),
                bins,
                average_as_profile,
            ).reshape(radial_grid.shape)
        else:
            raise ValueError(
                f"interpolation_mode = {interpolation_mode} not supported."
            )
        return average_as_profile, average_as_grid
    else:
        return average_as_profile
