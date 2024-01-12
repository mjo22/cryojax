"""
Utilities for cryojax modules that can be stacked along leading axes and indexed.
"""

from __future__ import annotations

__all__ = ["StackedModule", "StackedType"]

import math
import dataclasses
from typing import Type, TypeVar
from typing_extensions import Self
from jaxtyping import Array

import equinox as eqx
import jax.numpy as jnp

from ._field import field


StackedType = TypeVar("StackedType", bound="StackedModule")


class StackedModule(eqx.Module):
    """
    A ``Module`` whose Arrays are stacked along leading axes.

    This Module adds utilities for working with vmap.
    """

    n_stacked_dims: int = field(static=True, default=0, kw_only=True)

    def __check_init__(self):
        if self.n_stacked_dims < 0:
            raise ValueError(
                "Number of stacked axes must be greater than zero."
            )

    @classmethod
    def from_list(
        cls: Type[StackedType], modules: list[StackedType]
    ) -> StackedType:
        """
        Stack a list of StackedModules along a leading axis.
        """
        n_stacked_dims = modules[0].n_stacked_dims
        if not all([cls == type(obj) for obj in modules]):
            raise TypeError(
                f"Objects in the stack should all be of type {cls}."
            )
        if not all([n_stacked_dims == obj.n_stacked_dims for obj in modules]):
            raise TypeError(
                f"Objects to stack should all have the same number of stacked dimensions."
            )
        # Gather static and traced fields separately
        other, stacked = {}, {}
        for field in dataclasses.fields(modules[0]):
            name = field.name
            if name == "n_stacked_dims":
                pass
            elif isinstance(getattr(modules[0], name), Array):
                # Arrays get stacked.
                stacked[name] = jnp.stack(
                    [getattr(density, name) for density in modules], axis=0
                )
            else:
                # Static fields or Modules should all match, so take the first.
                other[name] = getattr(modules[0], name)
        return cls(**stacked, **other, n_stacked_dims=n_stacked_dims + 1)

    @property
    def stack_shape(self) -> tuple[int, ...]:
        if self.n_stacked_dims > 0:
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if isinstance(value, Array):
                    return value.shape[0 : self.n_stacked_dims]
            raise AttributeError(
                f"Could not get the stack_shape of the {type(self)}."
            )
        else:
            return ()

    def __len__(self) -> int:
        return math.prod(self.stack_shape)

    def __getitem__(self, idx) -> Self:
        if self.n_stacked_dims > 0:
            indexed = {}
            n_stacked_dims = self.n_stacked_dims
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if isinstance(value, Array):
                    # ... index all arrays in the stack
                    indexed[field.name] = value[idx]
                    n_stacked_dims = self.n_stacked_dims - (
                        value.ndim - indexed[field.name].ndim
                    )
            return dataclasses.replace(
                self, **indexed, n_stacked_dims=n_stacked_dims
            )
        else:
            raise IndexError(
                f"Tried to index a {type(self)} with n_stacked_dims = 0."
            )
