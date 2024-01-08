"""
Core functionality in cryojax, i.e. base classes and metadata.
"""

from __future__ import annotations

__all__ = ["field", "Module", "BufferModule"]

from typing import Any
from jaxtyping import Array

import jax
import jax.numpy as jnp

import equinox as eqx


def field(
    *,
    encode: Any = Array,
    **kwargs: Any,
) -> Any:
    """
    Add default metadata to usual dataclass fields through python
    and equinox.
    """
    # Equinox metadata
    static = kwargs.pop("static", False)
    if static:
        _converter = lambda x: x
    else:
        # This converter is necessary when a parameter is typed as,
        # for example, Optional[Real_].
        _converter = (
            lambda x: jnp.asarray(x) if isinstance(x, ArrayLike) else x
        )
    converter = kwargs.pop("converter", _converter)

    return eqx.field(
        converter=converter,
        static=static,
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
