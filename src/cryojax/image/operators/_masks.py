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
    """
    Base class for computing and applying an image mask.
    """

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


class CustomMask(AbstractImageMultiplier, strict=True):
    """
    Pass a custom mask as an array.
    """

    buffer: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]

    def __init__(
        self, mask: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ):
        self.buffer = mask


class CircularMask(AbstractMask, strict=True):
    """
    Apply a circular mask to an image.

    See documentation for
    ``cryojax.simulator.compute_circular_mask``
    for more information.

    Attributes
    ----------
    radius :
        The radius of the mask in Angstroms.
    rolloff :
        By default, ``0.05``.
    """

    buffer: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]

    radius: float = field(static=True)
    rolloff: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        radius: float,
        rolloff: float = 0.05,
    ) -> None:
        self.radius = radius
        self.rolloff = rolloff
        self.buffer = _compute_circular_mask(
            coordinate_grid_in_angstroms, self.radius, self.rolloff
        )


@overload
def _compute_circular_mask(
    coordinate_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    radius: float,
    rolloff: float,
) -> Float[Array, "y_dim x_dim"]: ...


@overload
def _compute_circular_mask(
    coordinate_grid_in_angstroms: Float[Array, "z_dim y_dim x_dim 3"],
    radius: float,
    rolloff: float,
) -> Float[Array, "z_dim y_dim x_dim"]: ...


def _compute_circular_mask(
    coordinate_grid_in_angstroms: (
        Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
    ),
    radius: float,
    rolloff: float = 0.05,
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    """
    Create a circular mask.

    Parameters
    ----------
    coordinate_grid :
        The image coordinates.
    grid_spacing :
        The grid spacing of ``coordinate_grid``.
    cutoff :
        The cutoff radius as a fraction of half
        the smallest box dimension. By default, ``0.95``.
    rolloff :
        The rolloff width as a fraction of the smallest box dimension.
        By default, ``0.05``.

    Returns
    -------
    mask : `Array`, shape `shape`
        An array representing the circular mask.
    """

    coords_norm = jnp.linalg.norm(coordinate_grid_in_angstroms, axis=-1)
    r_cut = radius

    coords_cut = coords_norm > r_cut

    rolloff_width = rolloff * coords_norm.max()
    mask = 0.5 * (
        1 + jnp.cos((coords_norm - r_cut - rolloff_width) / rolloff_width * jnp.pi)
    )

    mask = jnp.where(coords_cut, 0.0, mask)
    mask = jnp.where(coords_norm <= r_cut - rolloff_width, 1.0, mask)

    return mask
