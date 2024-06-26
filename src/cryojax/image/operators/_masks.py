"""
Masks to apply to images in real space.
"""

from typing import overload

import jax
import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Float

from ._operator import AbstractImageMultiplier


class AbstractMask(AbstractImageMultiplier, strict=True):
    """Base class for computing and applying an image mask."""

    @overload
    def __call__(
        self, image: Float[Array, "y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(
        self, image: Float[Array, "z_dim y_dim x_dim"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    def __call__(
        self, image: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        return image * jax.lax.stop_gradient(self.buffer)


class CustomMask(AbstractMask, strict=True):
    """Pass a custom mask as an array."""

    buffer: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]

    def __init__(
        self, mask: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ):
        self.buffer = mask


class CircularCosineMask(AbstractMask, strict=True):
    """Apply a circular mask to an image with a cosine
    soft-edge.
    """

    buffer: Float[Array, "y_dim x_dim"]

    radius_in_angstroms_or_pixels: float = field(static=True)
    rolloff_width_in_angstroms_or_pixels: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms_or_pixels: Float[Array, "y_dim x_dim 2"],
        radius_in_angstroms_or_pixels: float,
        rolloff_width_in_angstroms_or_pixels: float,
    ):
        """**Arguments:**

        - `coordinate_grid_in_angstroms_or_pixels`:
            The image coordinates.
        - `grid_spacing`:
            The pixel or voxel size of `coordinate_grid_in_angstroms_or_pixels`.
        - `radius_in_angstroms_or_pixels`:
            The radius of the circular mask.
        - `rolloff_width_in_angstroms_or_pixels`:
            The rolloff width of the soft edge.
        """
        self.radius_in_angstroms_or_pixels = radius_in_angstroms_or_pixels
        self.rolloff_width_in_angstroms_or_pixels = rolloff_width_in_angstroms_or_pixels
        self.buffer = _compute_circular_or_spherical_mask(
            coordinate_grid_in_angstroms_or_pixels,
            self.radius_in_angstroms_or_pixels,
            self.rolloff_width_in_angstroms_or_pixels,
        )


class SphericalCosineMask(AbstractMask, strict=True):
    """Apply a spherical mask to a volume with a cosine
    soft-edge.
    """

    buffer: Float[Array, "z_dim y_dim x_dim"]

    radius_in_angstroms_or_voxels: float = field(static=True)
    rolloff_width_in_angstroms_or_pixels: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms_or_voxels: Float[Array, "z_dim y_dim x_dim 3"],
        radius_in_angstroms_or_voxels: float,
        rolloff_width_in_angstroms_or_pixels: float,
    ):
        """**Arguments:**

        - `coordinate_grid_in_angstroms_or_voxels`:
            The volume coordinates.
        - `grid_spacing`:
            The pixel or voxel size of `coordinate_grid_in_angstroms_or_voxels`.
        - `radius_in_angstroms_or_voxels`:
            The radius of the spherical mask.
        - `rolloff_width_in_angstroms_or_pixels`:
            The rolloff width of the soft edge.
        """
        self.radius_in_angstroms_or_voxels = radius_in_angstroms_or_voxels
        self.rolloff_width_in_angstroms_or_pixels = rolloff_width_in_angstroms_or_pixels
        self.buffer = _compute_circular_or_spherical_mask(
            coordinate_grid_in_angstroms_or_voxels,
            self.radius_in_angstroms_or_voxels,
            self.rolloff_width_in_angstroms_or_pixels,
        )


@overload
def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"],
    radius: float,
    rolloff_width: float,
) -> Float[Array, "y_dim x_dim"]: ...


@overload
def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "z_dim y_dim x_dim 3"],
    radius: float,
    rolloff_width: float,
) -> Float[Array, "z_dim y_dim x_dim"]: ...


def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    radius: float,
    rolloff_width: float,
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    radial_coordinate_grid = jnp.linalg.norm(coordinate_grid, axis=-1)

    def compute_mask_at_coordinate(radial_coordinate):
        return jnp.where(
            radial_coordinate <= radius,
            1.0,
            jnp.where(
                radial_coordinate > radius + rolloff_width,
                0.0,
                0.5
                * (
                    1
                    + jnp.cos(jnp.pi * (radial_coordinate_grid - radius) / rolloff_width)
                ),
            ),
        )

    compute_mask = (
        jax.vmap(jax.vmap(compute_mask_at_coordinate))
        if radial_coordinate_grid.ndim == 2
        else jax.vmap(jax.vmap(jax.vmap(compute_mask_at_coordinate)))
    )

    return compute_mask(radial_coordinate_grid)
