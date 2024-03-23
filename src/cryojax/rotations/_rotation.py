"""
Abstraction of a rotation.
"""

from abc import abstractmethod
from typing import overload, Type
from typing_extensions import Self

from equinox import AbstractClassVar, Module
from jaxtyping import Array, PRNGKeyArray


class AbstractRotation(Module, strict=True):
    """Base class for a rotation."""

    space_dimension: AbstractClassVar[int]

    @overload
    def __matmul__(self, other: Self) -> Self: ...

    @overload
    def __matmul__(self, other: Array) -> Array: ...

    def __matmul__(self, other: Self | Array) -> Self | Array:
        """The `@` operator.

        Can either compose with other `AbstractRotation`s or
        act on arrays.
        """
        if isinstance(other, Array):
            return self.apply(other)
        elif isinstance(other, AbstractRotation):
            if not self.space_dimension == other.space_dimension:
                raise ValueError("Cannot compose rotations of different types.")
            return self.compose(other)
        else:
            raise ValueError(
                "Allowed types for `@` operator are arrays and other"
                f"rotations. Got {type(other)}."
            )

    @abstractmethod
    def apply(self, target: Array) -> Array:
        """Apply the rotation to an array."""
        raise NotImplementedError

    @abstractmethod
    def compose(self, other: Self) -> Self:
        """Composes this transformation with another."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self) -> Self:
        """Get the inverse of this rotation."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def identity(cls: Type[Self]) -> Self:
        """Return the identity element."""
        return NotADirectoryError

    @classmethod
    @abstractmethod
    def sample_uniform(cls: Type[Self], key: PRNGKeyArray) -> Self:
        """Draw a uniform sample."""
        raise NotImplementedError
