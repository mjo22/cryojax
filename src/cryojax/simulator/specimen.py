"""
Abstractions of biological specimen.
"""

from __future__ import annotations

__all__ = ["Specimen", "SpecimenMixture"]

from typing import Any
from functools import partial

from .scattering import ScatteringConfig
from .density import ElectronDensity
from .pose import Pose
from .conformation import Discrete
from ..core import Parameter, Array, dataclass, field, CryojaxObject


@partial(dataclass, kw_only=True)
class Specimen(CryojaxObject):
    """
    Abstraction of a biological specimen.

    Attributes
    ----------
    density : `cryojax.simulator.ElectronDensity`
        The electron density representation of the
        specimen.
    resolution : `cryojax.core.Parameter`
        Rasterization resolution.
        This is in dimensions of length.
    """

    resolution: Parameter = field()
    density: ElectronDensity = field()

    def scatter(
        self, scattering: ScatteringConfig, pose: Pose, **kwargs: Any
    ) -> Array:
        """
        Compute the scattered wave of the specimen in the
        exit plane.

        Arguments
        ---------
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration.
        pose : `cryojax.simulator.Pose`
            The imaging pose.
        """
        freqs = scattering.padded_freqs / self.resolution
        # View the electron density map at a given pose
        density = self.density.view(pose, **kwargs)
        # Compute the scattering image
        scattering_image = density.scatter(
            scattering, self.resolution, **kwargs
        )
        # Apply translation
        scattering_image = pose.shift(scattering_image, freqs)

        return scattering_image

    @property
    def draw(self) -> ElectronDensity:
        """Get the electron density."""
        return self.density


@dataclass
class SpecimenMixture(CryojaxObject):
    """
    A biological specimen at a mixture of conformations.

    Attributes
    ----------
    density : `list[cryojax.simulator.ElectronDensity]`
        The electron density representation of the
        specimen.
    conformation : `cryojax.simulator.Discrete`
        The conformational variable at which to evaulate
        the electron density.
    """

    density: list[ElectronDensity] = field()
    conformation: Discrete = field(default_factory=Discrete)

    @property
    def draw(self) -> ElectronDensity:
        """Draw the electron density at the configured conformation."""
        coordinate = self.conformation.coordinate
        if not (-len(coordinate) <= coordinate < len(coordinate)):
            raise ValueError("The conformational coordinate is out-of-bounds.")
        return self.density[coordinate]
