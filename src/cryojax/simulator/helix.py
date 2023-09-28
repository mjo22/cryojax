"""
Abstractions of helical filaments.
"""

from __future__ import annotations

__all__ = ["Helix", "compute_lattice"]

from typing import Any, Union, Optional, Callable
from jaxtyping import Array, Float, Int

import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as jtu
import equinox as eqx

from .specimen import Specimen
from .pose import Pose
from .optics import Optics
from .scattering import ScatteringConfig

from ..core import field, Module
from ..types import Real_, RealVector, ComplexImage

Lattice = Float[Array, "N 3"]
"""Type hint for array where each element is a lattice coordinate."""

Conformations = Union[Float[np.ndarray, "N"], Int[np.ndarray, "N"]]
"""Type hint for array where each element updates a Conformation."""


class Helix(Module):
    """
    Abstraction of a helical filament.

    This class acts just like a ``Specimen``, however
    it assembles a helix from a subunit.

    Attributes
    ----------
    subunit :
        The helical subunit.
    rise :
        The helical rise. This has dimensions
        of length.
    twist :
        The helical twist, given in degrees if
        ``degrees = True`` and radians otherwise.
    radius :
        The radius of the helix.
    conformations :
        The conformation of `subunit` at each lattice site.
        This can either be a fixed set of conformations or a function
        that computes conformations based on the lattice positions.
        In either case, the `Array` should be shape `(n_start*n_subunits,)`.
    n_start :
        The start number of the helix. By default, ``1``.
    n_subunits :
        The number of subunits in the assembly.
    degrees :
        Whether or not the helical twist is given in
        degrees. By default, ``True``.
    lattice : `Lattice`
        The 3D cartesian lattice coordinates for each subunit.
    """

    subunit: Specimen = field()
    rise: Union[Real_, RealVector] = field()
    twist: Union[Real_, RealVector] = field()
    radius: Union[Real_, RealVector] = field(default=1)
    conformations: Optional[
        Union[Conformations, Callable[[Lattice], Conformations]]
    ] = field(default=None)

    n_start: int = field(static=True, default=1)
    n_subunits: Optional[int] = field(static=True, default=None)
    degrees: bool = field(static=True, default=True)

    lattice: Lattice = field(static=True, init=False)

    def __post_init__(self):
        self.lattice = compute_lattice(
            self.rise,
            self.twist,
            radius=self.radius,
            n_start=self.n_start,
            n_subunits=self.n_subunits,
            degrees=self.degrees,
        )

    def scatter(
        self,
        scattering: ScatteringConfig,
        pose: Pose,
        optics: Optional[Optics] = None,
        **kwargs: Any,
    ) -> ComplexImage:
        """
        Compute the scattered wave of the specimen in the
        exit plane.

        The input and output of this method should identically
        match that of ``Specimen.scatter``.

        Arguments
        ---------
        scattering :
            The scattering configuration for the subunit.
        pose :
            The center of mass imaging pose of the helix.
        optics :
            The instrument optics.
        """
        image = self.subunit.scatter(scattering, pose, optics=optics, **kwargs)

        return image

    @property
    def resolution(self) -> Real_:
        """Hack to make this class act like a Specimen."""
        return self.subunit.resolution

    def draw(self) -> list[Specimen]:
        """Draw a realization of all of the subunits"""
        if (
            not hasattr(self.subunit, "conformation")
            or self.conformations is None
        ):
            return self.n_subunits * [self.subunit]
        else:
            if isinstance(self.conformations, Callable):
                c = self.conformations(self.lattice)
            else:
                c = self.conformations
            get_leaf = lambda subunit: subunit.conformation.coordinate
            return jtu.tree_map(
                lambda c: eqx.tree_at(get_leaf, self.subunit, c), c.tolist()
            )


def compute_lattice(
    rise: Union[Real_, RealVector],
    twist: Union[Real_, RealVector],
    radius: Union[Real_, RealVector] = 1.0,
    n_start: int = 1,
    n_subunits: Optional[int] = None,
    *,
    degrees: bool = True,
) -> Lattice:
    """
    Compute the lattice points of a helix for a given
    rise, twist, radius, and start number.

    Real_s
    ----------
    rise : `Real_` or `RealVector`, shape `(n_subunits,)`
        The helical rise.
    twist : `Real_` or `RealVector`, shape `(n_subunits,)`
        The helical twist.
    radius : `Real_` or `RealVector`, shape `(n_subunits,)`
        The radius of the helix.
    n_start :
        The start number of the helix.
    n_subunits :
        The number of subunits in the assembly for
        a single helix. The total number of subunits
        is really equal to ``n_start * n_subunits``.
        By default, ``2 * jnp.pi / twist``.
    degrees :
        Whether or not the angular Real_s
        are given in degrees or radians.

    Returns
    -------
    lattice : shape `(n_start*n_subunits, 3)`
        The helical lattice.
    """
    # Convert to radians
    if degrees:
        twist = jnp.deg2rad(twist)
    # If the number of subunits is not given, compute for one helix
    if n_subunits is None:
        n_subunits = int(2 * jnp.pi / twist)
    # Rotational symmetry between helices due to the start number
    symmetry_angles = jnp.array(
        [2 * jnp.pi * n / n_start for n in range(n_start)]
    )

    def compute_helical_coordinates(symmetry_angle):
        """
        Get the coordinates for a given
        helix, where the x and y coordinates
        are rotated by an angle.
        """
        theta = jnp.arange(n_subunits, dtype=float) * twist
        x_0 = radius * jnp.cos(theta)
        y_0 = radius * jnp.sin(theta)
        c, s = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R = jnp.array(((c, s), (-s, c)), dtype=float)
        x, y = R @ jnp.stack((x_0, y_0))
        z = rise * jnp.arange(-n_subunits / 2, n_subunits / 2, 1, dtype=float)
        return jnp.stack((x, y, z), axis=-1)

    # The helical coordinates for all sub-helices
    lattice = jax.vmap(compute_helical_coordinates)(symmetry_angles)
    return lattice.reshape((n_start * n_subunits, 3))
