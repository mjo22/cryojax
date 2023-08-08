"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["ParameterDict", "ParameterState"]

import dataclasses
from typing import TypedDict, Optional

from jax import random

from ..core import dataclass, field, Scalar, Serializable
from .pose import EulerPose
from .optics import NullOptics
from .exposure import UniformExposure
from .noise import NullNoise
from . import Pose, Optics, Exposure, Noise


class ParameterDict(TypedDict):
    """
    Dictionary to facilitate JAX functional
    transformations.
    """

    # Pose parameters
    offset_x: Optional[Scalar]
    offset_y: Optional[Scalar]

    view_phi: Optional[Scalar]
    view_theta: Optional[Scalar]
    view_psi: Optional[Scalar]

    view_qw: Optional[Scalar]
    view_qx: Optional[Scalar]
    view_qy: Optional[Scalar]
    view_qz: Optional[Scalar]

    # CTF parameters
    defocus_u: Optional[Scalar]
    defocus_v: Optional[Scalar]
    defocus_angle: Optional[Scalar]
    voltage: Optional[Scalar]
    spherical_aberration: Optional[Scalar]
    amplitude_contrast_ratio: Optional[Scalar]
    phase_shift: Optional[Scalar]
    b_factor: Optional[Scalar]

    # Noise parameters
    sigma: Optional[Scalar]
    kappa: Optional[Scalar]
    xi: Optional[Scalar]

    # Image intensity
    N: Optional[Scalar]
    mu: Optional[Scalar]


@dataclass
class ParameterState(Serializable):
    """
    PyTree container for the state of an ``ImageModel``.

    Attributes
    ----------
    pose : `cryojax.simulator.Pose`
        The image pose.
    optics : `cryojax.simulator.OpticsModel`
        The CTF model.
    exposure : `cryojax.simulator.Exposure`
        The model for intensity scaling.
    noise : ``cryojax.simulator.Noise``
        The noise model.
    """

    pose: Pose = field(default=EulerPose(), encode=Pose)
    optics: Optics = field(default=NullOptics(), encode=Optics)
    exposure: Exposure = field(default=UniformExposure())
    noise: Noise = field(
        default=NullNoise(key=random.PRNGKey(seed=0)), encode=Noise
    )

    def update(self, params: ParameterDict) -> ParameterState:
        """Return a new ParameterState based on a ParameterDict."""
        fields = dataclasses.fields(self)
        update = {}
        for field in fields:
            update[field.name] = {
                k: v
                for k, v in params.items()
                if hasattr(getattr(self, field.name), k)
            }
        return self.replace(
            pose=self.pose.replace(**update["pose"]),
            optics=self.optics.replace(**update["optics"]),
            exposure=self.exposure.replace(**update["exposure"]),
            noise=self.noise.replace(**update["noise"]),
        )
