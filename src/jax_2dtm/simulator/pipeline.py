"""
Routines to return a cryo-EM image rendering pipeline
"""

__all__ = ["get_rendering_pipeline", "get_log_likelihood"]


from .coordinates import Cloud
from .transform import rotate_and_translate, Pose
from .image import project, ImageConfig


def get_rendering_pipeline(config: ImageConfig, cloud: Cloud):
    """
    Return function handle for the imaging pipeline.
    """

    def pipeline():
        pass

    return pipeline


def get_log_likelihood():
    """
    Return function handle for the log likelihood.
    """
    pass
