"""
Core functionality in cryojax, i.e. base classes and metadata.
"""

from __future__ import annotations

__all__ = ["field", "Module", "BufferModule"]

from typing import Any
from jaxtyping import Array, ArrayLike

import jax
import jax.numpy as jnp

import equinox as eqx


def field(
    *,
    stack: bool = False,
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
    # ... add the stack keyword
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
