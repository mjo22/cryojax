"""
Core functionality in cryojax, i.e. base classes and metadata.
"""

from __future__ import annotations

__all__ = ["field", "Module", "BatchedModule", "BufferModule"]

import math
import dataclasses
from typing import Any, Type, TypeVar
from typing_extensions import Self
from jaxtyping import Array, ArrayLike

import jax
import jax.numpy as jnp

import equinox as eqx

_T = TypeVar("_T", bound="Module")


def field(
    *,
    stack: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Add default metadata to usual dataclass fields through dataclasses
    and equinox.

    Arguments
    ---------
    stack : `bool`
        Metadata that indicates if a field is to be stacked
        when giving Module fields a batch dimension. This metadata
        is not currently supported in every cryojax Module, and is
        only for internal cryojax batch dimension functionality.
        In general, a user can add batch dimensions however they want.

        As a rule of thumb, this should only be set to ``False`` if the
        field in question is cumbersome and unecessary to store. See
        ``cryojax.simulator.density.VoxelGrid`` for an example. This
        argument is not used if ``static = True``.
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
    # ... add the stack keyword, if the field is traced
    if not static:
        metadata["stack"] = stack

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


class BatchedModule(Module):
    """
    A ``Module`` with batching utiltiies.

    Note that a ``BatchedModule`` must directly
    have jax arrays as its dataclass fields.
    """

    n_batch_dims: int = field(static=True, default=0, kw_only=True)

    def __check_init__(self):
        if self.n_batch_dims < 0:
            raise ValueError(
                "Number of batching axes must be greater than zero."
            )

    @classmethod
    def from_stack(cls: Type[_T], stack: list[_T]) -> _T:
        """
        Stack a list of electron densities along the leading
        axis of a single electron density.
        """
        if not all([cls == type(obj) for obj in stack]):
            raise TypeError(
                f"Objects in the stack should all be of type {cls}."
            )
        # Gather static and traced fields separately
        other, stacked = {}, {}
        for field in dataclasses.fields(stack[0]):
            name = field.name
            if name == "n_batch_dims":
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
        return cls(**stacked, **other, n_batch_dims=1)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        if self.n_batch_dims > 0:
            for field in dataclasses.fields(self):
                if not (
                    ("static" in field.metadata and field.metadata["static"])
                    or (
                        "stack" in field.metadata
                        and not field.metadata["stack"]
                    )
                ):
                    value = getattr(self, field.name)
                    return value.shape[0 : self.n_batch_dims]
            raise AttributeError(
                f"Could not get the batch_shape of the {type(self)}."
            )
        else:
            return ()

    def __len__(self) -> int:
        return math.prod(self.batch_shape)

    def __getitem__(self, idx) -> Self:
        if self.n_batch_dims > 0:
            # Gather static and traced fields separately
            indexed = {}
            for field in dataclasses.fields(self):
                name = field.name
                if not (
                    ("static" in field.metadata and field.metadata["static"])
                    or (
                        "stack" in field.metadata
                        and not field.metadata["stack"]
                    )
                ):
                    # Get stacked fields at particular index
                    indexed[name] = getattr(self, name)[idx]
            return dataclasses.replace(self, **indexed, n_batch_dims=0)
        else:
            raise IndexError(
                f"Tried to index a {type(self)} with n_batch_dims = 0."
            )


class BufferModule(Module):
    """
    A Module composed of buffers (do not take gradients).
    """

    def __getattribute__(self, __name: str) -> Any:
        value = super().__getattribute__(__name)
        if isinstance(value, Array):
            return jax.lax.stop_gradient(value)
        else:
            return value
