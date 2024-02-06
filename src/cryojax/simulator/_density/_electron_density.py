"""
Base electron density representation.
"""

from abc import abstractmethod
from typing import Any, TypeVar
from typing_extensions import Self
from jaxtyping import PyTree
from equinox import Module

from .._pose import AbstractPose
from ...coordinates import get_not_coordinate_filter_spec


ElectronDensityT = TypeVar("ElectronDensityT", bound="AbstractElectronDensity")
"""TypeVar for an electron density."""


def is_density_leaves_without_coordinates(element: Any) -> bool | PyTree[bool]:
    """Returns a filter spec that is ``True`` at the ``ElectronDensity``
    leaves, besides its coordinates.
    """
    if isinstance(element, AbstractElectronDensity):
        return get_not_coordinate_filter_spec(element)
    else:
        return False


class AbstractElectronDensity(Module, strict=True):
    """
    Abstraction of an electron density distribution.
    """

    @abstractmethod
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """
        View the electron density at a given pose.

        In subclasses, fourier-space electron density representations
        should rotate coordinates by a backrotation (the inverse rotation).

        Arguments
        ---------
        pose :
            The imaging pose.
        """
        raise NotImplementedError
