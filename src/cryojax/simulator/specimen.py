"""
Abstractions of biological specimen.
"""

from __future__ import annotations

__all__ = ["Specimen", "Ensemble"]

from typing import Any, Optional

from .scattering import ScatteringConfig
from .density import ElectronDensity
from .exposure import Exposure
from .pose import Pose, EulerPose
from .ice import Ice, NullIce
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
        the electron density. This does not do anything in
        the specimen base class and should be overwritten
        in subclasses.
    pose :
        The pose of the specimen.
    """

    density: ElectronDensity = field()
    resolution: Real_ = field()
    conformation: Any = field(default=None)

    pose: Pose = field(default_factory=EulerPose)

    def scatter(
        self,
        scattering: ScatteringConfig,
        pose: Optional[Pose] = None,
        exposure: Optional[Exposure] = None,
        optics: Optional[Optics] = None,
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
        pose = pose or self.pose
        freqs = scattering.padded_freqs / self.resolution
        # Draw the electron density at a particular conformation
        density = self.sample()
        # View the electron density map at a given pose
        density = density.view(pose)
        # Compute the scattering image
        image = scattering.scatter(density, self.resolution)
        # Apply translation
        image *= pose.shifts(freqs)
        # Compute and apply CTF
        if optics is not None:
            ctf = optics(freqs, pose=pose)
            image = optics.apply(ctf, image)
        # Apply the electron exposure model
        if exposure is not None:
            scaling, offset = exposure.scaling(freqs), exposure.offset(freqs)
            image = scaling * image + offset

        return image

    def sample(self) -> ElectronDensity:
        """Get the electron density."""
        return self.density


class Ensemble(Specimen):
    """
    A biological specimen at a discrete mixture of conformations.
    """

    density: list[ElectronDensity] = field()
    conformation: Discrete = field(default_factory=Discrete)

    def __check_init__(self):
        coordinate = self.conformation.coordinate
        if not (-len(self.density) <= coordinate < len(self.density)):
            raise ValueError("The conformational coordinate is out-of-bounds.")

    def sample(self) -> ElectronDensity:
        """Sample the electron density at the configured conformation."""
        return self.density[self.conformation.coordinate]
