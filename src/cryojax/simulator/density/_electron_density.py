"""
Base electron density representation.
"""

__all__ = ["ElectronDensity"]

from abc import abstractmethod
from typing import Optional, Type

from equinox import AbstractVar

from ..pose import Pose
from ...core import Module, field


class ElectronDensity(Module):
    """
    Abstraction of an electron density map.

    Attributes
    ----------
    is_real :
        Whether or not the representation is
        real or fourier space.
    """

    is_real: AbstractVar[bool]
    _is_stacked: bool = field(static=True, default=False, kw_only=True)

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
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_stack(
        cls: Type["ElectronDensity"], stack: list["ElectronDensity"]
    ) -> "ElectronDensity":
        """
        Stack a list of electron densities along the leading
        axis of a single electron density.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> "ElectronDensity":
        """Get a particular electron density in the stack."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of electron densities in the stack."""
        raise NotImplementedError
