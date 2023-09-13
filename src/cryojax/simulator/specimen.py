"""
Routines for representing and operating on 3D point clouds.
"""

from __future__ import annotations

__all__ = ["Specimen", "Helix"]

from typing import Any

from .scattering import ScatteringConfig
from .density import ElectronDensity
from .pose import Pose
from ..core import Array, dataclass, field, CryojaxObject


@dataclass
class Specimen(CryojaxObject):
    """
    Abstraction of a biological specimen.

    Attributes
    ----------
    density : `cryojax.simulator.ElectronDensity`
        The electron density representation of the
        specimen.
    """

    density: ElectronDensity = field()

    def view(self, pose: Pose, **kwargs: Any) -> Specimen:
        """
        View the specimen at the given pose.

        Arguments
        ---------
        pose : `cryojax.simulator.Pose`
            The imaging pose.
        """
        density = self.density.view(pose, **kwargs)

        return self.replace(density=density)

    def scatter(self, scattering: ScatteringConfig, **kwargs: Any) -> Array:
        """
        Compute the scattered wave of the specimen in the
        exit plane.

        Arguments
        ---------
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration.
        """
        return self.density.scatter(scattering, **kwargs)


@dataclass
class Helix(Specimen):
    """
    Abstraction of a helical filament.

    Attributes
    ----------
    density : `cryojax.simulator.ElectronDensity`
        The electron density representation of the
        helical subunit.
    """

    def view(self, pose: Pose, **kwargs: Any) -> Specimen:
        """
        View the specimen at the given pose.

        Arguments
        ---------
        pose : `cryojax.simulator.Pose`
            The imaging pose.
        """
        raise NotImplementedError

    def scatter(self, scattering: ScatteringConfig, **kwargs: Any) -> Array:
        """
        Compute the scattered wave of the specimen in the
        exit plane.

        Arguments
        ---------
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration.
        """
        raise NotImplementedError
