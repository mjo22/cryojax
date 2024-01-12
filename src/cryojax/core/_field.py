"""
Wrapper for the for eqx.field.
"""

from __future__ import annotations

__all__ = ["field"]

from typing import Any, Optional, Callable
from jaxtyping import ArrayLike

import equinox as eqx
import jax.numpy as jnp


def field(
    converter: Optional[Callable] = None, static: bool = False, **kwargs: Any
) -> Any:
    """Wrap eqx.field to add a default converter."""
    # Equinox metadata
    if converter is None:
        if static:
            return eqx.field(static=True, **kwargs)
        else:
            default_converter = (
                lambda x: jnp.asarray(x) if isinstance(x, ArrayLike) else x
            )
            return eqx.field(
                static=False, converter=default_converter, **kwargs
            )
    else:
        return eqx.field(static=static, converter=converter, **kwargs)
