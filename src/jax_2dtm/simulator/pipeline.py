"""
Routines to return a cryo-EM image rendering pipeline
"""

__all__ = ["RenderingPipeline"]


import dataclasses
from typing import Optional
from .cloud import Cloud
from .image import ImageConfig, ImageModel
from .state import ParameterState
from ..types import Array


@dataclasses.dataclass
class RenderingPipeline:
    """ """

    model: ImageModel
    config: ImageConfig
    cloud: Cloud
    observed: Optional[Array] = None

    def __call__(self, params: ParameterState):
        return self.model(
            params, self.config, self.cloud, observed=self.observed
        )
