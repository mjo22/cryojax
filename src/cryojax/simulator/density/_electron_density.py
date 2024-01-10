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
from ...core import Module, field


ElectronDensityType = TypeVar("ElectronDensityType", bound="ElectronDensity")


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

    @classmethod
    def from_stack(
        cls: Type[ElectronDensityType], stack: list[ElectronDensityType]
    ) -> ElectronDensityType:
        """
        Stack a list of electron densities along the leading
        axis of a single electron density.
        """
        if not all([cls == type(density) for density in stack]):
            raise TypeError(
                "Electron density stack should all be of the same type."
            )
        # Gather static and traced fields separately
        other, stacked = {}, {}
        for field in dataclasses.fields(stack[0]):
            name = field.name
            if name == "_is_stacked":
                pass
            elif ("static" in field.metadata and field.metadata["static"]) or (
                "stack" in field.metadata and not field.metadata["stack"]
            ):
                # Static or unstacked fields should all match, so take the first.
                other[name] = getattr(stack[0], name)
            else:
                # Traced fields, unless specified in metadata, get stacked.
                stacked[name] = jnp.stack(
                    [getattr(density, name) for density in stack], axis=0
                )
        return cls(**stacked, **other, _is_stacked=True)

    def __getitem__(self, idx: int) -> Self:
        if self._is_stacked:
            # Gather static and traced fields separately
            indexed = {}
            for field in dataclasses.fields(self):
                name = field.name
                if name == "_is_stacked":
                    pass
                elif (
                    "static" in field.metadata and field.metadata["static"]
                ) or (
                    "stack" in field.metadata and not field.metadata["stack"]
                ):
                    pass
                else:
                    # Get stacked fields at particular index
                    indexed[name] = getattr(self, name)[idx]
            return dataclasses.replace(self, **indexed, _is_stacked=False)
        else:
            raise IndexError("Cannot index an non-stacked ElectronDensity.")

    def __len__(self) -> int:
        if self._is_stacked:
            for field in dataclasses.fields(self):
                if not (
                    ("static" in field.metadata and field.metadata["static"])
                    or (
                        "stack" in field.metadata
                        and not field.metadata["stack"]
                    )
                ):
                    return getattr(self, field.name).shape[0]
            raise AttributeError(
                "Could not get the length of the ElectronDensity stack."
            )
        else:
            raise TypeError(
                "Cannot get length of non-stacked ElectronDensity."
            )
