"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = ["ScatteringModel"]

from abc import abstractmethod

import jax.numpy as jnp

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

    Methods
    -------
    scatter:
        The scattering model.
    """

    manager: ImageManager = field()

    @abstractmethod
    def scatter(
        self, density: ElectronDensity, resolution: Real_
    ) -> ComplexImage:
        """
        Compute the scattered wave of the electron
        density in the exit plane.

        Arguments
        ---------
        density :
            The electron density representation.
        resolution :
            The rasterization resolution.
        """
        raise NotImplementedError

    def _normalize(self, image: ComplexImage) -> ComplexImage:
        """Normalize images on the exit plane according to cisTEM conventions"""
        M1, M2 = image.shape
        # Set zero frequency component to zero
        image = image.at[0, 0].set(0.0 + 0.0j)
        # cisTEM normalization convention for projections
        return image / jnp.sqrt(M1 * M2)
