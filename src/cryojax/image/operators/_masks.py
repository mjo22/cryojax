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
    rolloff_width_fraction: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms_or_pixels: Float[Array, "y_dim x_dim 2"],
        radius_in_angstroms_or_pixels: float,
        rolloff_width_fraction: float = 0.05,
    ):
        """**Arguments:**

        - `coordinate_grid_in_angstroms_or_pixels`:
            The image coordinates.
        - `grid_spacing`:
            The pixel or voxel size of `coordinate_grid_in_angstroms_or_pixels`.
        - `radius_in_angstroms_or_pixels`:
            The radius of the circular mask.
        - `rolloff_width_fraction`:
            The rolloff width as a fraction of the smallest box dimension.
            By default, ``0.05``.
        """
        self.radius_in_angstroms_or_pixels = radius_in_angstroms_or_pixels
        self.rolloff_width_fraction = rolloff_width_fraction
        self.buffer = _compute_circular_or_spherical_mask(
            coordinate_grid_in_angstroms_or_pixels,
            self.radius_in_angstroms_or_pixels,
            self.rolloff_width_fraction,
        )


class SphericalCosineMask(AbstractMask, strict=True):
    """Apply a spherical mask to a volume with a cosine
    soft-edge.
    """

    buffer: Float[Array, "z_dim y_dim x_dim"]

    radius_in_angstroms_or_voxels: float = field(static=True)
    rolloff_width_fraction: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms_or_voxels: Float[Array, "z_dim y_dim x_dim 3"],
        radius_in_angstroms_or_voxels: float,
        rolloff_width_fraction: float = 0.05,
    ):
        """**Arguments:**

        - `coordinate_grid_in_angstroms_or_voxels`:
            The volume coordinates.
        - `grid_spacing`:
            The pixel or voxel size of `coordinate_grid_in_angstroms_or_voxels`.
        - `radius_in_angstroms_or_voxels`:
            The radius of the spherical mask.
        - `rolloff_width_fraction`:
            The rolloff width as a fraction of the smallest box dimension.
            By default, ``0.05``.
        """
        self.radius_in_angstroms_or_voxels = radius_in_angstroms_or_voxels
        self.rolloff_width_fraction = rolloff_width_fraction
        self.buffer = _compute_circular_or_spherical_mask(
            coordinate_grid_in_angstroms_or_voxels,
            self.radius_in_angstroms_or_voxels,
            self.rolloff_width_fraction,
        )


@overload
def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"],
    radius: float,
    rolloff: float,
) -> Float[Array, "y_dim x_dim"]: ...


@overload
def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "z_dim y_dim x_dim 3"],
    radius: float,
    rolloff: float,
) -> Float[Array, "z_dim y_dim x_dim"]: ...


def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    radius: float,
    rolloff: float = 0.05,
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    coords_norm = jnp.linalg.norm(coordinate_grid, axis=-1)
    r_cut = radius

    coords_cut = coords_norm > r_cut

    rolloff_width = rolloff * coords_norm.max()
    mask = 0.5 * (
        1 + jnp.cos((coords_norm - r_cut - rolloff_width) / rolloff_width * jnp.pi)
    )

    mask = jnp.where(coords_cut, 0.0, mask)
    mask = jnp.where(coords_norm <= r_cut - rolloff_width, 1.0, mask)

    return mask
