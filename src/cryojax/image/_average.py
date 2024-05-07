"""
Routines to compute radial averages of images.
"""

from functools import partial
from typing import overload

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact


@overload
def radial_average(
    image: Inexact[Array, "y_dim x_dim"],
    radial_grid: Float[Array, "y_dim x_dim"],
    bins: Float[Array, " n_bins"],
) -> Inexact[Array, " n_bins"]: ...


@overload
def radial_average(
    image: Inexact[Array, "z_dim y_dim x_dim"],
    radial_grid: Float[Array, "z_dim y_dim x_dim"],
    bins: Float[Array, " n_bins"],
) -> Inexact[Array, " n_bins"]: ...


@overload
def radial_average(
    image: Inexact[Array, "y_dim x_dim"],
    radial_grid: Float[Array, "y_dim x_dim"],
    bins: Float[Array, " n_bins"],
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
) -> tuple[Inexact[Array, " n_bins"], Inexact[Array, "y_dim x_dim"]]: ...


@overload
def radial_average(
    image: Inexact[Array, "z_dim y_dim x_dim"],
    radial_grid: Float[Array, "z_dim y_dim x_dim"],
    bins: Float[Array, " n_bins"],
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
) -> tuple[Inexact[Array, " n_bins"], Inexact[Array, "z_dim y_dim x_dim"]]: ...


@partial(jax.jit, static_argnames=["to_grid", "interpolation_mode"])
def radial_average(
    image: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    radial_grid: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"],
    bins: Float[Array, " n_bins"],
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
) -> (
    Inexact[Array, " n_bins"]
    | tuple[
        Inexact[Array, " n_bins"],
        Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    ]
):
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
            raise ValueError(f"interpolation_mode = {interpolation_mode} not supported.")
        return average_as_profile, average_as_grid
    else:
        return average_as_profile
