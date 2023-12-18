"""
Base electron density representation.
"""

__all__ = ["ElectronDensity"]

from abc import abstractmethod
from typing import Optional, Type

from equinox import AbstractVar

from ..pose import Pose
from ...core import Module


class ElectronDensity(Module):
    """
    Abstraction of an electron density map.

    Attributes
    ----------
    real :
        Whether or not the representation is
        real or fourier space.
    """

    real: AbstractVar[bool]

    @abstractmethod
    def rotate_to(self, pose: Pose) -> "ElectronDensity":
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
        cls: Type["ElectronDensity"],
        filename: str,
        config: Optional[dict] = None,
    ) -> "ElectronDensity":
        """
        Load an ElectronDensity from a file.

        This method should be used to instantiate and
        deserialize ElectronDensity.
        """
        raise NotImplementedError
