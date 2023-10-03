"""
Abstractions of helical filaments.
"""

from __future__ import annotations

__all__ = ["Helix", "compute_lattice"]

from typing import Any, Union, Optional, Callable
from jaxtyping import Array, Float, Int
from functools import cached_property

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from .specimen import Specimen
from .exposure import Exposure
from .pose import Pose
from .optics import Optics
from .scattering import ScatteringConfig

from ..core import field, Module
from ..types import Real_, RealVector, ComplexImage

Lattice = Float[Array, "N 3"]
"""Type hint for array where each element is a lattice coordinate."""

Conformations = Union[Float[Array, "N"], Int[Array, "N"]]
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

    def scatter(
        self,
        scattering: ScatteringConfig,
        pose: Pose,
        exposure: Optional[Exposure] = None,
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
        exposure :
            The exposure model.
        optics :
            The instrument optics.
        """
        # Draw the conformations of each subunit
        subunits = self.subunits
        # Compute the pose of each subunit
        where = lambda p: (p.offset_x, p.offset_y, p.offset_z)
        transformed_lattice = pose.rotate(self.lattice) + pose.offset
        poses = jtu.tree_map(
            lambda r: eqx.tree_at(where, pose, (r[0, 0], r[0, 1], r[0, 2])),
            jnp.split(transformed_lattice, len(subunits), axis=0),
        )
        # Compute all projection images
        scatter = lambda s, p: s.scatter(
            scattering, pose=p, exposure=None, optics=optics, **kwargs
        )
        images = jtu.tree_map(
            scatter, subunits, poses, is_leaf=lambda s: isinstance(s, Specimen)
        )
        # Sum them all together
        image = jtu.tree_reduce(lambda x, y: x + y, images)
        # Apply the electron exposure model
        if exposure is not None:
            freqs = scattering.padded_freqs / self.resolution
            scaling, offset = exposure.scaling(freqs), exposure.offset(freqs)
            image = scaling * image + offset

        return image

    @property
    def resolution(self) -> Real_:
        """Hack to make this class act like a Specimen."""
        return self.subunit.resolution

    @cached_property
    def lattice(self) -> Lattice:
        """Get the helical lattice."""
        return compute_lattice(
            self.rise,
            self.twist,
            radius=self.radius,
            n_start=self.n_start,
            n_subunits=self.n_subunits,
            degrees=self.degrees,
        )

    @cached_property
    def subunits(self) -> list[Specimen]:
        """Draw a realization of all of the subunits"""
        if (
            not hasattr(self.subunit, "conformation")
            or self.conformations is None
        ):
            return self.n_subunits * self.n_start * [self.subunit]
        else:
            if isinstance(self.conformations, Callable):
                cs = self.conformations(self.lattice)
            else:
                cs = self.conformations
            where = lambda s: s.conformation.coordinate
            return jax.lax.map(
                lambda c: eqx.tree_at(where, self.subunit, c), cs
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

    Arguments
    ---------
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
        Whether or not the angular parameters
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
