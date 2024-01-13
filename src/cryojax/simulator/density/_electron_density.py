"""
Base electron density representation.
"""

__all__ = ["is_density_leaves", "ElectronDensity", "ElectronDensityType"]

from abc import abstractmethod
from typing import Type, Any, TypeVar
from typing_extensions import Self
from jaxtyping import PyTree
from equinox import AbstractClassVar

from ..pose import Pose
from ...core import IndexedModule
from ...image import get_not_coordinate_filter_spec


ElectronDensityType = TypeVar("ElectronDensityType", bound="ElectronDensity")


def is_density_leaves(element: Any) -> bool | PyTree[bool]:
    """Returns a filter spec that is ``True`` at the ``ElectronDensity``
    leaves, besides its coordinates.
    """
    if isinstance(element, ElectronDensity):
        return get_not_coordinate_filter_spec(element)
    else:
        return False


class ElectronDensity(IndexedModule):
    """
    Abstraction of an electron density distribution.

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
