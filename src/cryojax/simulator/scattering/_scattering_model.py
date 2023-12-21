"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = ["ScatteringModel"]

from abc import abstractmethod

from ..density import ElectronDensity
from ..manager import ImageManager

from ...core import field, Module
from ...typing import Real_, ComplexImage


class ScatteringModel(Module):
    """
    A model of electron scattering onto the exit plane of the specimen.

    In subclasses, overwrite the ``ScatteringConfig.scatter``
    routine.

    Attributes
    ----------
    manager:
        Handles image configuration and
        utility routines.
    pixel_size :
        Rasterization pixel size. This is in
        dimensions of length.

    Methods
    -------
    scatter:
        The scattering model.
    """

    manager: ImageManager = field()
    pixel_size: Real_ = field()

    @abstractmethod
    def scatter(self, density: ElectronDensity) -> ComplexImage:
        """
        Compute the scattered wave of the electron
        density in the exit plane.

        Arguments
        ---------
        density :
            The electron density representation.
        """
        raise NotImplementedError
