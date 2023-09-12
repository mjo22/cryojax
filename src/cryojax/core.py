"""
Core functionality such as type hinting, dataclasses, and serialization.

See https://jax.readthedocs.io/en/latest/jax.typing.html for jax
type hint conventions.
"""

from __future__ import annotations

__all__ = [
    "Array",
    "ArrayLike",
    "Parameter",
    "ParameterDict",
    "dataclass",
    "field",
    "Serializable",
    "CryojaxObject",
]


import base64
import dataclasses
import jax
from typing import (
    Annotated,
    Any,
    Callable,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp
import numpy as np

from dataclasses_json import DataClassJsonMixin, config
from dataclasses_json.mm import JsonData

Float = Union[float, Annotated[Array, (), jnp.floating]]
"""Type alias for Union[float, Annotated[Array, (), jnp.floating]]"""

Parameter = Union[float, Annotated[Array, (), jnp.floating]]
"""Type alias for Union[float, Annotated[Array, (), jnp.floating]]"""

ParameterDict = dict[str, Parameter]
"""Type alias for dict[str, Parameter]"""


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
def dataclass(clz: Type[Any], **kwargs: Any) -> Type[Any]:
    data_clz: Any = dataclasses.dataclass(frozen=True, **kwargs)(clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(name)
        else:
            meta_fields.append(name)
    meta_fields = tuple(meta_fields)
    data_fields = tuple(data_fields)

    def replace(self: Any, **updates: _T) -> _T:
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    def iterate_clz(x: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        data = tuple(getattr(x, name) for name in data_fields)
        meta = tuple(getattr(x, name) for name in meta_fields)
        data_args = tuple(zip(data_fields, data))
        meta_args = tuple(zip(meta_fields, meta))
        return data_args, meta_args

    def iterate_data(x: Any) -> Tuple[Any, ...]:
        data = tuple(getattr(x, name) for name in data_fields)
        return data

    def iterate_meta(x: Any) -> Tuple[Any, ...]:
        meta = tuple(getattr(x, name) for name in meta_fields)
        return meta

    def clz_from_iterable(
        meta_args: Tuple[Any, ...], data_args: Tuple[Any, ...]
    ) -> Any:
        kwargs = dict(meta_args + data_args)
        return data_clz(**kwargs)

    jax.tree_util.register_pytree_with_keys(
        data_clz, iterate_clz, clz_from_iterable
    )

    # Hack to make this class act as a tuple when unpacked
    data_clz.iter_data = lambda self: iterate_data(self)
    data_clz.iter_meta = lambda self: iterate_meta(self)

    @property
    def data(self) -> tuple[tuple, tuple]:
        """Return data of the model."""
        return data_fields, iterate_data(self)

    @property
    def metadata(self) -> tuple[tuple, tuple]:
        """Return metadata of the model."""
        return meta_fields, iterate_meta(self)

    data_clz.data = data
    data_clz.metadata = metadata

    return data_clz


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
        Type hint for the field's json encoding. If
        `False`, do not encode the field.
    """
    # Inspect kwargs
    if "metadata" in kwargs.keys():
        metadata = kwargs["metadata"]
        del kwargs["metadata"]
    else:
        metadata = {}
    if "init" in kwargs.keys():
        init = kwargs["init"]
    else:
        init = True
    metadata.update(dict(pytree_node=pytree_node, encode=encode))
    # Set serialization metadata
    if init:
        if encode is False:
            serializer = config(decoder=_dummy_decoder, encoder=_dummy_encoder)
        elif encode == Array:
            serializer = config(encoder=_np_encoder, decoder=_jax_decoder)
        elif encode == np.ndarray:
            serializer = config(encoder=_np_encoder, decoder=_np_decoder)
        else:
            serializer = config(
                encoder=_cryojax_encoder, decoder=_cryojax_decoder
            )
    else:
        serializer = config(decoder=_dummy_decoder, encoder=_dummy_encoder)

    metadata.update(serializer)
    return dataclasses.field(metadata=metadata, **kwargs)


# This section implements serialization functionality for cryojax
# objects. This subclasses DataClassJsonMixin from dataclasses-json
# and provides custom encoding/decoding for Arrays and cryojax
# objects.
@dataclass
class Serializable(DataClassJsonMixin):
    """
    Base class for serializable ``cryojax`` dataclasses.
    """

    @classmethod
    def load(cls, filename: str, **kwargs: Any) -> Serializable:
        """
        Load a ``cryojax`` object from a file.
        """
        with open(filename, "r", encoding="utf-8") as f:
            s = f.read()
        return cls.from_json(s, **kwargs)

    def dump(self, filename: str, **kwargs: Any) -> Serializable:
        """
        Dump a ``cryojax`` object to a file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.to_json(**kwargs))

    @classmethod
    def loads(cls, s: JsonData, **kwargs: Any) -> Serializable:
        """
        Load a ``cryojax`` object from a json string.
        """
        return cls.from_json(s, **kwargs)

    def dumps(self, **kwargs: Any) -> JsonData:
        """
        Dump a ``cryojax`` object to a json string.
        """
        return self.to_json(**kwargs)


@dataclass
class CryojaxObject(Serializable):
    """
    Base class for ``cryojax`` dataclasses.
    """

    def update(self, **params: ParameterDict) -> CryojaxObject:
        """
        Return a new CryojaxObject based on a dictionary.

        If ``params`` contains any pytree nodes in this instance,
        they will be updated. Nested ``CryojaxObject``s are
        supported.

        Note that the update will fail for nodes with identical
        names.
        """
        keys = params.keys()
        nleaves = len(keys)
        if nleaves > 0:
            updates = {}
            names, fields = self.data
            for idx in range(len(names)):
                name, field = names[idx], fields[idx]
                if name in keys:
                    updates[name] = params[name]
                elif isinstance(field, CryojaxObject):
                    updates[name] = field.update(**params)
            return self.replace(**updates)
        else:
            return self


def _dummy_encoder(x: Any) -> str:
    """Encode nothing"""
    pass


def _dummy_decoder(x: Any) -> Any:
    """Decode nothing"""
    pass


def _np_encoder(x: Any) -> Any:
    """Numpy array encoder"""
    if isinstance(x, (np.ndarray, Array)):
        x = np.asarray(x)
        data_b64 = base64.b64encode(np.ascontiguousarray(x).data)
        return dict(
            __ndarray__=data_b64.decode("ascii"),
            dtype=str(x.dtype),
            shape=x.shape,
        )
    elif isinstance(x, complex):
        return dict(__complex__=True, real=x.real, imag=x.imag)
    else:
        return _cryojax_encoder(x)


def _np_decoder(x: Any) -> Any:
    """Numpy array decoder"""
    if isinstance(x, dict) and "__ndarray__" in x:
        data = base64.b64decode(x["__ndarray__"])
        return np.frombuffer(data, x["dtype"]).reshape(x["shape"])
    elif isinstance(x, dict) and "__complex__" in x:
        return complex(x["real"], x["imag"])
    else:
        return _cryojax_decoder(x)


def _jax_decoder(x: Any) -> Any:
    """Jax array decoder"""
    a = _np_decoder(x)
    if isinstance(a, np.ndarray):
        return jnp.asarray(a)
    else:
        return a


def _cryojax_encoder(x: Any) -> Any:
    """Encode a CryojaxObject or a list of them."""
    if isinstance(x, CryojaxObject):
        return dict(
            __name__=x.__class__.__name__, __dataclass_fields__=x.to_dict()
        )
    elif isinstance(x, list):
        return [_cryojax_encoder(xi) for xi in x]
    else:
        return x


def _cryojax_decoder(x: Any) -> Any:
    """Decode a CryojaxObject or a list of them."""
    if isinstance(x, dict) and "__name__" in x:
        from . import simulator

        cls = getattr(simulator, x["__name__"])
        return cls.from_dict(x["__dataclass_fields__"])
    elif isinstance(x, list):
        return [_cryojax_decoder(xi) for xi in x]
    else:
        return x
