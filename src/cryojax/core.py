"""
Core functionality, such as type hints and base classes.
"""

from __future__ import annotations

__all__ = ["field", "Module"]

from types import FunctionType
from typing import Any, Union
from jaxtyping import Array, ArrayLike, Float, Complex, Int

import jax.numpy as jnp
import numpy as np

import dataclasses
import equinox as eqx

import marshal
import base64
from dataclasses_json import DataClassJsonMixin, config
from dataclasses_json.mm import JsonData


def field(
    *,
    encode: Any = Array,
    **kwargs: Any,
) -> Any:
    """
    Add metadata to usual dataclass fields.

    Parameters
    ----------
    encode : `Any`
        Type hint for the field's json encoding. If
        `False`, do not encode the field.
    """
    # Dataclass kwargs
    metadata = kwargs.pop("metadata", {})
    init = kwargs.pop("init", True)
    # Equinox kwargs
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
    # Get serialization metadata
    if init:
        if encode is False:
            encoder = config(decoder=_dummy_decoder, encoder=_dummy_encoder)
        elif encode == Array:
            encoder = config(encoder=_np_encoder, decoder=_jax_decoder)
        elif encode == np.ndarray:
            encoder = config(encoder=_np_encoder, decoder=_np_decoder)
        else:
            encoder = config(encoder=_object_encoder, decoder=_object_decoder)
    else:
        encoder = config(decoder=_dummy_decoder, encoder=_dummy_encoder)
    # Update metadata for serialization
    metadata["encode"] = encode
    metadata.update(encoder)
    # Update metadata for equinox
    metadata["converter"] = converter
    if static:
        metadata["static"] = True

    # FIXME: We really should be wrapping eqx.field.
    # For some reason, when metadata is passed as a
    # keyword to eqx.field, converter and static
    # are not added to the metadata.

    return dataclasses.field(
        metadata=metadata,
        init=init,
        **kwargs,
    )


class _Serializable(DataClassJsonMixin):
    """
    Base class for serializable ``cryojax`` dataclasses.

    This class implements serialization functionality for cryojax
    objects. This subclasses DataClassJsonMixin from dataclasses-json
    and provides custom encoding/decoding for Arrays and cryojax
    objects.
    """

    @classmethod
    def load(cls, filename: str, **kwargs: Any) -> _Serializable:
        """
        Load a ``cryojax`` object from a file.
        """
        with open(filename, "r", encoding="utf-8") as f:
            s = f.read()
        return cls.from_json(s, **kwargs)

    def dump(self, filename: str, **kwargs: Any):
        """
        Dump a ``cryojax`` object to a file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.to_json(**kwargs))

    @classmethod
    def loads(cls, s: JsonData, **kwargs: Any) -> _Serializable:
        """
        Load a ``cryojax`` object from a json string.
        """
        return cls.from_json(s, **kwargs)

    def dumps(self, **kwargs: Any) -> JsonData:
        """
        Dump a ``cryojax`` object to a json string.
        """
        return self.to_json(**kwargs)


class Module(eqx.Module, _Serializable):
    """
    Base class for ``cryojax`` objects.
    """


# 0-d array type hints
Real_ = Float[Array, ""]
"""Type hint for a real-valued number."""

Complex_ = Complex[Array, ""]
"""Type hint for an integer."""

Integer_ = Int[Array, ""]
"""Type hint for an integer."""

# 1-d array type hints
RealVector = Float[Array, "N"]
"""Type hint for a real-valued vector."""

ComplexVector = Complex[Array, "N"]
"""Type hint for an complex-valued vector."""

Vector = Union[RealVector, ComplexVector]
"""Type hint for a vector."""

# 2-d array type hints
RealImage = Float[Array, "N1 N2"]
"""Type hint for an real-valued image."""

ComplexImage = Complex[Array, "N1 N2"]
"""Type hint for an complex-valued image."""

Image = Union[RealImage, ComplexImage]
"""Type hint for an image."""

ImageCoords = Float[Array, "N1 N2 2"]
"""Type hint for a coordinate system."""

# 3-d array type hints
RealVolume = Float[Array, "N1 N2 N3"]
"""Type hint for an real-valued volume."""

ComplexVolume = Complex[Array, "N1 N2 N3"]
"""Type hint for an complex-valued volume."""

Volume = Union[RealVolume, ComplexVolume]
"""Type hint for an volume."""

VolumeCoords = Float[Array, "N1 N2 N3 3"]
"""Type hint for a volume coordinate system."""

# 3D Point cloud type hints (non-uniformly spaced points).
RealCloud = Float[Array, "N"]
"""Type hint for a real-valued point cloud."""

ComplexCloud = Complex[Array, "N"]
"""Type hint for a complex-valued point cloud."""

Cloud = Union[RealCloud, ComplexCloud]
"""Type hint for a point cloud."""

CloudCoords = Float[Array, "N 2"]
"""Type hint for a point cloud coordinate system."""


#
# Encoders
#
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
            __ndarray__=data_b64.decode(),
            dtype=str(x.dtype),
            shape=x.shape,
        )
    else:
        return _object_encoder(x)


def _np_decoder(x: Any) -> Any:
    """Numpy array decoder"""
    if isinstance(x, dict) and "__ndarray__" in x:
        data = base64.b64decode(x["__ndarray__"])
        return np.frombuffer(data, x["dtype"]).reshape(x["shape"])
    else:
        return _object_decoder(x)


def _jax_decoder(x: Any) -> Any:
    """Jax array decoder"""
    a = _np_decoder(x)
    if isinstance(a, np.ndarray):
        return jnp.asarray(a)
    else:
        return a


def _object_encoder(x: Any) -> Any:
    """Encoder for python objects, or collections of them."""
    if isinstance(x, Module):
        return _cryojax_encoder(x)
    elif isinstance(x, FunctionType):
        return _function_encoder(x)
    elif isinstance(x, complex):
        return _complex_encoder(x)
    elif isinstance(x, list):
        return [_object_encoder(xi) for xi in x]
    else:
        return x


def _object_decoder(x: Any) -> Any:
    """Decoder for python objects, or collections of them."""
    if isinstance(x, dict) and "__class__" in x:
        return _cryojax_decoder(x)
    if isinstance(x, dict) and "__code__" in x:
        return _function_decoder(x)
    elif isinstance(x, dict) and "__complex__" in x:
        return _complex_decoder(x)
    elif isinstance(x, list):
        return [_object_decoder(xi) for xi in x]
    else:
        return x


def _cryojax_encoder(x: Module) -> dict[str, Union[str, dict]]:
    """Encode a Module"""
    return dict(__class__=x.__class__.__name__, __dict__=x.to_dict())


def _cryojax_decoder(x: dict[str, Union[str, dict]]) -> Module:
    """Decode a Module"""
    from . import simulator

    cls = getattr(simulator, x["__class__"])
    return cls.from_dict(x["__dict__"])


def _function_encoder(x: FunctionType) -> dict[str, str]:
    """Encode a FunctionType"""
    return dict(
        __name__=x.__name__,
        __code__=marshal.dumps(x.__code__).decode("raw_unicode_escape"),
    )


def _function_decoder(x: dict[str, str]) -> FunctionType:
    """Decode a FunctionType"""
    return FunctionType(
        marshal.loads(bytes(x["__code__"], "raw_unicode_escape")),
        globals(),
        x["__name__"],
    )


def _complex_encoder(x: complex) -> dict[str, Union[str, float]]:
    """Encode a complex number"""
    return dict(__complex__=True, real=x.real, imag=x.imag)


def _complex_decoder(x: dict[str, Union[str, float]]) -> complex:
    """Decode a complex number"""
    return complex(x["real"], x["imag"])
