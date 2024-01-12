"""
Core functionality in cryojax, i.e. base classes and metadata.
"""

from __future__ import annotations

__all__ = [
    "field",
    "Module",
    "StackedModule",
    "CoordinateList",
    "CoordinateGrid",
    "FrequencyGrid",
]

import math
import dataclasses
from typing import Any, Type, TypeVar
from typing_extensions import Self
from jaxtyping import Array, ArrayLike
from .typing import ImageCoords, VolumeCoords, CloudCoords2D, CloudCoords3D

import jax
import jax.numpy as jnp

import equinox as eqx

_T = TypeVar("_T", bound="Module")


def field(**kwargs: Any) -> Any:
    """
    Add default metadata to usual dataclass fields through dataclasses
    and equinox.
    """
    # Equinox metadata
    static = kwargs.pop("static", False)
    if static:
        # ... null converter
        _converter = lambda x: x
    else:
        # ... this converter is necessary when a parameter is typed as,
        # for example, Optional[Real_].
        _converter = (
            lambda x: jnp.asarray(x) if isinstance(x, ArrayLike) else x
        )
    # ... set the converter to the passed converter or the default convertere
    converter = kwargs.pop("converter", _converter)
    # Cryojax metadata
    metadata = kwargs.pop("metadata", {})

    return eqx.field(
        converter=converter,
        static=static,
        metadata=metadata,
        **kwargs,
    )


class Module(eqx.Module):
    """
    Base class for ``cryojax`` objects.
    """


class CoordinateList(eqx.Module):
    """
    A Pytree that wraps a coordinate list.
    """

    _coordinates: CloudCoords3D | CloudCoords2D = field(converter=jnp.asarray)

    def get(self):
        """Get the coordinate list."""
        return self._coordinates

    def __mul__(self: CoordinateList, arr: ArrayLike) -> CoordinateList:
        return CoordinateList(self._coordinates * jnp.asarray(arr))

    def __rmul__(self: CoordinateList, arr: ArrayLike) -> CoordinateList:
        return CoordinateList(jnp.asarray(arr) * self._coordinates)

    def __truediv__(self: CoordinateList, arr: ArrayLike) -> CoordinateList:
        return CoordinateList(self._coordinates / jnp.asarray(arr))

    def __rtruediv__(self: CoordinateList, arr: ArrayLike) -> CoordinateList:
        return CoordinateList(jnp.asarray(arr) / self._coordinates)


class CoordinateGrid(eqx.Module):
    """
    A Pytree that wraps a coordinate grid.
    """

    _coordinates: ImageCoords | VolumeCoords = field(converter=jnp.asarray)

    def get(self):
        """Get the coordinate grid."""
        return jax.lax.stop_gradient(self._coordinates)

    def __mul__(self: CoordinateGrid, arr: ArrayLike) -> CoordinateGrid:
        return CoordinateGrid(self._coordinates * jnp.asarray(arr))

    def __rmul__(self: CoordinateGrid, arr: ArrayLike) -> CoordinateGrid:
        return CoordinateGrid(jnp.asarray(arr) * self._coordinates)

    def __truediv__(self: CoordinateGrid, arr: ArrayLike) -> CoordinateGrid:
        return CoordinateGrid(self._coordinates / jnp.asarray(arr))

    def __rtruediv__(self: CoordinateGrid, arr: ArrayLike) -> CoordinateGrid:
        return CoordinateGrid(jnp.asarray(arr) / self._coordinates)


FrequencyGrid = CoordinateGrid


class StackedModule(Module):
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
    def from_list(cls: Type[_T], modules: list[_T]) -> _T:
        """
        Stack a list of electron densities along the leading
        axis of a single electron density.
        """
        if not all([cls == type(obj) for obj in modules]):
            raise TypeError(
                f"Objects in the stack should all be of type {cls}."
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
        return cls(**stacked, **other, n_stacked_dims=1)

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
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if isinstance(value, Array):
                    # ... index all arrays in the stack
                    indexed[field.name] = value[idx]
            return dataclasses.replace(self, **indexed, n_stacked_dims=0)
        else:
            raise IndexError(
                f"Tried to index a {type(self)} with n_stacked_dims = 0."
            )
