"""
Core functionality such as type hinting, dataclasses, and serialization.

See https://jax.readthedocs.io/en/latest/jax.typing.html for jax
type hint conventions.
"""

__all__ = [
    "Array",
    "ArrayLike",
    "Scalar",
    "dataclass",
    "field",
    "Serializable",
]


import dataclasses
import jax
from typing import (
    Any,
    Callable,
    Tuple,
    Type,
    TypeVar,
    Union,
    _UnionGenericAlias,
    get_args,
    get_origin,
)
from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp
import numpy as np
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


# This section implements serialization functionality for jax-2dtm
# objects. This subclasses DataClassJsonMixin from dataclasses-json
# and provides custom encoding/decoding for Arrays and jax-2dtm
# objects.
@dataclass
class Serializable(DataClassJsonMixin):
    """
    Base class for serializable ``jax-2dtm`` dataclasses.
    """


def np_encoder(x: Any) -> Any:
    """Encoder for jax arrays and datatypes."""
    if isinstance(x, Array):
        return np.array(x).tolist()
    elif isinstance(x, np.generic):
        return x.item()
    else:
        return x


def np_decoder(x: Any) -> Any:
    """Decode list to jax array."""
    return np.asarray(x) if isinstance(x, list) else x


def jax_decoder(x: Any) -> Any:
    """Decode list to jax array."""
    return jnp.asarray(x) if isinstance(x, list) else x


def union_decoder(x: Any, union: _UnionGenericAlias) -> Any:
    """Decode a union type hint."""
    instance = None
    for cls in get_args(union):
        try:
            temp = cls.from_dict(x)
            assert set(x.keys()) == set(temp.to_dict().keys())
            instance = temp
        except (KeyError, TypeError, AssertionError):
            pass
    if instance is None:
        raise TypeError(f"Could not decode from {union}")
    return instance


def field(
    pytree_node: bool = True,
    encode: Any = Array,
    **kwargs: Any,
) -> Any:
    """
    Add metadata to usual dataclass fields.

    Parameters
    ----------
    pytree_node : `bool`
        Determine if field is to be part of the
        pytree.
    encode : `Any`
        Type hint for the field's json encoding.
        If this is a ``Union`` of ``jax_2dtm``
        objects, the decoder will try to find
        the correct one to instantiate.
    """
    metadata = dict(pytree_node=pytree_node)
    if get_origin(encode) is Union:
        serializer = config(decoder=lambda x: union_decoder(x, encode))
    elif encode == Array:
        serializer = config(encoder=np_encoder, decoder=jax_decoder)
    elif encode == np.ndarray:
        serializer = config(encoder=np_encoder, decoder=np_decoder)
    else:
        serializer = {}
    metadata.update(serializer)
    return dataclasses.field(metadata=metadata, **kwargs)
