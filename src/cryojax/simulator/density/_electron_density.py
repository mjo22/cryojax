"""
Base electron density representation.
"""

__all__ = ["ElectronDensity"]

import dataclasses
from abc import abstractmethod
from typing import Optional, Type
from equinox import AbstractClassVar

import jax.numpy as jnp

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

    is_real: AbstractClassVar[bool]
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
    def from_stack(
        cls: Type["ElectronDensity"], stack: list["ElectronDensity"]
    ) -> "ElectronDensity":
        """
        Stack a list of electron densities along the leading
        axis of a single electron density.
        """
        if not all([cls == type(density) for density in stack]):
            raise TypeError(
                "Electron density stack should all be of the same type."
            )
        if not all([stack[0].is_real == density.is_real for density in stack]):
            raise TypeError(
                "Electron density stack should all be in real or fourier space."
            )
        # Gather static and traced fields separately
        static, traced = {}, {}
        for field in dataclasses.fields(stack[0]):
            name = field.name
            if name == "_is_stacked":
                pass
            elif "static" in field.metadata and field.metadata["static"]:
                # Static fields should all match, so take the first.
                static[name] = getattr(stack[0], name)
            else:
                # Traced fields get stacked.
                traced[name] = jnp.stack(
                    [getattr(density, name) for density in stack], axis=0
                )
        return cls(**traced, **static, _is_stacked=True)

    def __getitem__(self, idx: int) -> "ElectronDensity":
        if self._is_stacked:
            cls = type(self)
            # Gather static and traced fields separately
            static, traced = {}, {}
            for field in dataclasses.fields(self):
                name = field.name
                if name == "_is_stacked":
                    pass
                elif "static" in field.metadata and field.metadata["static"]:
                    # Get static fields
                    static[name] = getattr(self, name)
                else:
                    # Get traced fields at particular index
                    traced[name] = getattr(self, name)[idx]
            return cls(**traced, **static, _is_stacked=False)
        else:
            return self

    def __len__(self) -> int:
        if self._is_stacked:
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if (
                    "static" not in field.metadata
                    or not field.metadata["static"]
                ):
                    return value.shape[0]
            raise AttributeError(
                "Could not get the length of the ElectronDensity stack."
            )
        else:
            return 1
