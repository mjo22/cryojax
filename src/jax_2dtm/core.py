"""
See https://jax.readthedocs.io/en/latest/jax.typing.html
"""

__all__ = ["Array", "ArrayLike", "Scalar", "dataclass", "Serializable"]


import dataclasses
import jax
from typing import Any, Callable, Tuple, Type, TypeVar, Union
from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp
import numpy as np
from jaxlib.xla_extension import ArrayImpl
from dataclasses_json import DataClassJsonMixin, config


Scalar = Union[float, Array]
"""Type alias for Union[float, Array]"""


# This section follows the implementation in tinygp, which is based closely on the
# implementation in flax:
#
# https://github.com/dfm/tinygp/blob/9dceb7f6fa09537022c9cd95be7b7f55350a0a06/src/tinygp/helpers.py
# https://github.com/google/flax/blob/b60f7f45b90f8fc42a88b1639c9cc88a40b298d3/flax/struct.py
#
# This decorator is interpreted by static analysis tools as a hint
# that a decorator or metaclass causes dataclass-like behavior.
# See https://github.com/microsoft/pyright/blob/main/specs/dataclass_transforms.md
# for more information about the __dataclass_transform__ magic.
_T = TypeVar("_T")


def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
) -> Callable[[_T], _T]:
    # If used within a stub file, the following implementation can be
    # replaced with "...".
    return lambda a: a


@__dataclass_transform__()
def dataclass(clz: Type[Any]) -> Type[Any]:
    data_clz: Any = dataclasses.dataclass(frozen=True)(clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(name)
        else:
            meta_fields.append(name)

    def replace(self: Any, **updates: _T) -> _T:
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    def iterate_clz(x: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    def clz_from_iterable(meta: Tuple[Any, ...], data: Tuple[Any, ...]) -> Any:
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return data_clz(**kwargs)

    jax.tree_util.register_pytree_node(
        data_clz, iterate_clz, clz_from_iterable
    )

    # Hack to make this class act as a tuple when unpacked
    data_clz.iter_data = lambda self: iterate_clz(self)[0].__iter__()
    data_clz.iter_meta = lambda self: iterate_clz(self)[1].__iter__()

    return data_clz


@dataclass
class Serializable(DataClassJsonMixin):
    """
    Base class for serializable ``jax-2dtm`` dataclasses.
    """


def ndarray_encoder(x):
    """Encode jax array as list"""
    return np.array(x).tolist() if type(x) is ArrayImpl else x


def ndarray_decoder(x):
    """Decode list to jax array"""
    return jnp.asarray(x) if type(x) is list else x


def field(pytree_node: bool = True, array=True, **kwargs: Any) -> Any:
    metadata = dict(pytree_node=pytree_node)
    if array:
        metadata.update(
            config(encoder=ndarray_encoder, decoder=ndarray_decoder)
        )
    return dataclasses.field(metadata=metadata, **kwargs)
