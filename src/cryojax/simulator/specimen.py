"""
Abstractions of biological specimen.
"""

from __future__ import annotations

__all__ = ["Specimen", "SpecimenMixture"]

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
    conformation: Any = field()

    def __init__(
        self,
        density: ElectronDensity,
        resolution: Real_,
        conformation: Optional[Any] = None,
    ):
        self.density = density
        self.resolution = resolution
        self.conformation = None

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
        density = self.sample()
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
            image = exposure.rescale(image, real=False)

        return image

    def sample(self) -> ElectronDensity:
        """Get the electron density."""
        return self.density


class SpecimenMixture(Specimen):
    """
    A biological specimen at a mixture of conformations.
    """

    density: list[ElectronDensity] = field()
    conformation: Discrete = field()

    def __init__(
        self,
        density: ElectronDensity,
        resolution: Real_,
        conformation: Optional[Discrete] = None,
    ):
        self.density = density
        self.resolution = resolution
        self.conformation = conformation or Discrete()

    def sample(self) -> ElectronDensity:
        """Sample the electron density at the configured conformation."""
        coordinate = self.conformation.coordinate
        if not (-len(coordinate) <= coordinate < len(coordinate)):
            raise ValueError("The conformational coordinate is out-of-bounds.")
        return self.density[coordinate]
