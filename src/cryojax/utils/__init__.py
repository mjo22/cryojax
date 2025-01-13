"""This is a module of utilities for working with JAX/Equinox. This includes utilities
for Equinox filtered transformations and Equinox recommendations for creating custom
per-leaf behavior for pytrees.
"""

from ._filter_specs import get_filter_spec as get_filter_spec
from ._filtered_transformations import (
    filter_grad_with_spec as filter_grad_with_spec,
    filter_value_and_grad_with_spec as filter_value_and_grad_with_spec,
    filter_vmap_with_spec as filter_vmap_with_spec,
)
from ._transforms import (
    AbstractPyTreeTransform as AbstractPyTreeTransform,
    CustomTransform as CustomTransform,
    resolve_transforms as resolve_transforms,
    StopGradientTransform as StopGradientTransform,
)
