"""
See https://jax.readthedocs.io/en/latest/jax.typing.html
"""

__all__ = ["Array", "ArrayLike", "Scalar"]


from jax import Array
from jax.typing import ArrayLike
from typing import Union


Scalar = Union[float, Array]
"""Type alias for Union[float, Array]"""
