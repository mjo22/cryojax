"""
Base electron density representation.
"""

__all__ = ["ElectronDensity", "ElectronDensityType"]

import dataclasses
from abc import abstractmethod
from typing import Type, Any, TypeVar
from typing_extensions import Self
from equinox import AbstractClassVar

import jax.numpy as jnp

from ..pose import Pose
from ...core import BatchedModule, field


ElectronDensityType = TypeVar("ElectronDensityType", bound="ElectronDensity")


class ElectronDensity(BatchedModule):
    """
    Abstraction of an electron density map.

    Attributes
    ----------
    is_real :
        Whether or not the representation is
        real or fourier space.
    """

    is_real: AbstractClassVar[bool]

    @abstractmethod
    def rotate_to_pose(self, pose: Pose) -> Self:
        """
        View the electron density at a given pose.

        Arguments
        ---------
        pose :
            The imaging pose.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(
        cls: Type[ElectronDensityType],
        filename: str,
        **kwargs: Any,
    ) -> ElectronDensityType:
        """
        Load an ElectronDensity from a file.
        """
        raise NotImplementedError
