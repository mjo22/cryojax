"""Pytrees that represent particle stacks."""

from equinox import AbstractVar, Module
from jaxtyping import Array, Inexact


class AbstractParticleStack(Module, strict=True):
    """An abstraction of a particle stack, represented by a pytree
    that stores an image stack.

    Subclasses should also include parameters included in the
    image formation model, typically represented with `cryojax` objects.
    """

    image_stack: AbstractVar[Inexact[Array, "... y_dim x_dim"]]
