"""
Abstractions of biological specimen.
"""

from __future__ import annotations

__all__ = ["Specimen", "SpecimenMixture"]

from dataclasses import KW_ONLY
from typing import Any, Optional

from .scattering import ScatteringConfig
from .density import ElectronDensity
from .exposure import Exposure
from .pose import Pose
from .optics import Optics
from .conformation import Discrete
from ..core import field, Module
from ..types import Real_, ComplexImage


class Specimen(Module):
    """
    Abstraction of a biological specimen.

    Attributes
    ----------
    density :
        The electron density representation of the
        specimen.
    resolution :
        Rasterization resolution. This is in
        dimensions of length.
    conformation :
        The conformational variable at which to evaulate
        the electron density. This should be overwritten
        in subclasses.
    """

    density: ElectronDensity = field()
    resolution: Real_ = field()
    conformation: Any = field(default=None)

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

        Arguments
        ---------
        scattering :
            The scattering configuration.
        pose :
            The imaging pose.
        exposure :
            The exposure model.
        optics :
            The instrument optics.
        """
        freqs = scattering.padded_freqs / self.resolution
        # Draw the electron density at a particular conformation
        density = self.draw()
        # View the electron density map at a given pose
        density = density.view(pose, **kwargs)
        # Compute the scattering image
        image = density.scatter(scattering, self.resolution, **kwargs)
        # Apply translation
        image = pose.shift(image, freqs)
        # Compute and apply CTF
        if optics is not None:
            ctf = optics(freqs, pose=pose)
            image = optics.apply(ctf, image)
        # Apply the electron exposure model
        if exposure is not None:
            image = exposure.scale(image, real=False)

        return image

    def draw(self) -> ElectronDensity:
        """Get the electron density."""
        return self.density


class SpecimenMixture(Module):
    """
    A biological specimen at a mixture of conformations.
    """

    density: list[ElectronDensity] = field()
    conformation: Discrete = field(default_factory=Discrete)

    def draw(self) -> ElectronDensity:
        """Draw the electron density at the configured conformation."""
        coordinate = self.conformation.coordinate
        if not (-len(coordinate) <= coordinate < len(coordinate)):
            raise ValueError("The conformational coordinate is out-of-bounds.")
        return self.density[coordinate]
