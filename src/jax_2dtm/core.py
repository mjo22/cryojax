"""
Core functionality for jax-2dtm objects.
"""

from __future__ import annotations

__all__ = ["Serializable"]

import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxlib.xla_extension import ArrayImpl

from dataclasses_json import DataClassJsonMixin
from dataclasses_json.core import (
    _asdict,
    _decode_dataclass,
    Json,
)  # This is a nasty hack!

from .types import dataclass


@dataclass
class Serializable(DataClassJsonMixin):
    """
    Base class for serializable ``jax-2dtm`` dataclasses.
    """

    @classmethod
    def from_dict(
        cls: type[Serializable], kvs: Json, *, infer_missing=False
    ) -> Serializable:
        f = lambda x: jnp.asarray(x) if type(x) is list else x
        is_leaf = lambda x: True if type(x) is list else False
        return _decode_dataclass(
            cls, tree_map(f, kvs, is_leaf=is_leaf), infer_missing
        )

    def to_dict(self, encode_json=False) -> dict[str, Json]:
        f = lambda x: np.array(x).tolist() if type(x) is ArrayImpl else x
        return tree_map(f, _asdict(self, encode_json=encode_json))
