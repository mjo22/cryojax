"""
Atomic-based electron density representations.
"""

from __future__ import annotations

__all__ = ["AtomCloud"]

from typing import Type, Any, ClassVar

import equinox as eqx
from jaxtyping import Array

from ._electron_density import AbstractElectronDensity
from ..pose import AbstractPose
from ...core import field


class AtomCloud(AbstractElectronDensity):
    """
    Abstraction of a point cloud of atoms.
    """

    weights: Array = field()
    coordinate_list: Array = field()
    variances: Array = field()
    identity: Array = field()

    is_real: ClassVar[bool] = True

    def rotate_to(self, pose: AbstractPose) -> AtomCloud:
        return eqx.tree_at(
            lambda d: d.coordinate_list,
            self,
            pose.rotate(self.coordinate_list, is_real=self.is_real),
        )

    @classmethod
    def from_file(
        cls: Type[AtomCloud],
        filename: str,
        **kwargs: Any,
    ) -> AtomCloud:
        """
        Load an Atom Cloud

        TODO: What is the file format appropriate here? Q. for Michael...
        """
        import gemmi

        raise NotImplementedError
        # return cls.from_mrc(filename, config=config, **kwargs)
