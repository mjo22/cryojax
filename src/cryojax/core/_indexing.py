"""
Utilities for cryojax modules that can be stacked along leading axes and indexed.
"""

from __future__ import annotations

__all__ = ["IndexedModule", "IndexedT"]

import math
import dataclasses
from typing import Type, TypeVar
from typing_extensions import Self
from jaxtyping import Array

import equinox as eqx
import jax.numpy as jnp

from ._field import field


IndexedT = TypeVar("IndexedT", bound="IndexedModule")


class IndexedModule(eqx.Module):
    """
    A ``Module`` whose ``Array``s are stacked along leading axes
    so that they can be indexed externally.
    """

    n_indexed_dims: int = field(static=True, default=0, kw_only=True)

    def __check_init__(self):
        if self.n_indexed_dims < 0:
            raise ValueError(
                "Number of indexed axes must be greater than zero."
            )

    @classmethod
    def from_list(cls: Type[IndexedT], modules: list[IndexedT]) -> IndexedT:
        """
        Stack a list of IndexedModules along a leading axis.
        """
        n_indexed_dims = modules[0].n_indexed_dims
        if not all([cls == type(obj) for obj in modules]):
            raise TypeError(
                f"Objects in the list should all be of type {cls}."
            )
        if not all([n_indexed_dims == obj.n_indexed_dims for obj in modules]):
            raise TypeError(
                f"Objects to list should all have the same number of indexed dimensions."
            )
        # Gather static and traced fields separately
        other, stacked = {}, {}
        for field in dataclasses.fields(modules[0]):
            name = field.name
            if name == "n_indexed_dims":
                pass
            elif isinstance(getattr(modules[0], name), Array):
                # Arrays get stacked.
                stacked[name] = jnp.stack(
                    [getattr(density, name) for density in modules], axis=0
                )
            else:
                # Static fields or Modules should all match, so take the first.
                other[name] = getattr(modules[0], name)
        return cls(**stacked, **other, n_indexed_dims=n_indexed_dims + 1)

    @property
    def stack_shape(self) -> tuple[int, ...]:
        if self.n_indexed_dims > 0:
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if isinstance(value, Array):
                    return value.shape[0 : self.n_indexed_dims]
            raise AttributeError(
                f"Could not get the stack_shape of the {type(self)}."
            )
        else:
            return ()

    def __len__(self) -> int:
        return math.prod(self.stack_shape)

    def __getitem__(self, idx) -> Self:
        if self.n_indexed_dims > 0:
            indexed = {}
            n_indexed_dims = self.n_indexed_dims
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if isinstance(value, Array):
                    # ... index all arrays in the stack
                    indexed[field.name] = value[idx]
                    n_indexed_dims = self.n_indexed_dims - (
                        value.ndim - indexed[field.name].ndim
                    )
            return dataclasses.replace(
                self, **indexed, n_indexed_dims=n_indexed_dims
            )
        else:
            raise IndexError(
                f"Tried to index a {type(self)} with n_indexed_dims = 0."
            )
