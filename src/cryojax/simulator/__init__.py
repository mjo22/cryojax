__all__ = [
    # Functional API
    "rotate_and_translate_rpy",
    "rotate_and_translate_wxyz",
    "project_with_nufft",
    "project_with_gaussians",
    # "project_with_slice",
    "compute_anti_aliasing_filter",
    "compute_whitening_filter",
    "compute_circular_mask",
    "compute_ctf_power",
    "rescale_image",
    # Image configuration
    "ImageConfig",
    "NufftScattering",
    "GaussianScattering",
    # "FourierSliceScattering"
    # Specimen representations
    "ElectronCloud",
    "ElectronGrid",
    # Filters
    "AntiAliasingFilter",
    "WhiteningFilter",
    # Masks
    "CircularMask",
    # Model parameter configuration
    "ParameterState",
    ## Pose
    "EulerPose",
    "QuaternionPose",
    ## Ice
    "EmpiricalIce",
    "ExponentialIce",
    ## Optics
    "CTFOptics",
    ## Electron dose
    "Dose",
    ## Detector models
    "WhiteDetector",
    # Image models
    "ScatteringImage",
    "OpticsImage",
    "DetectorImage",
    "GaussianImage",
    # Type hints
    "ScatteringConfig",
    "Pose",
    "Filter",
    "Optics",
    "Intensity",
    "Noise",
    "Image",
]

from typing import Union

from .pose import (
    rotate_and_translate_rpy,
    rotate_and_translate_wxyz,
    EulerPose,
    QuaternionPose,
)

Pose = Union[EulerPose, QuaternionPose]
from .scattering import (
    project_with_nufft,
    project_with_gaussians,
    ImageConfig,
    NufftScattering,
    GaussianScattering,
    # FourierSliceScattering,
)

ScatteringConfig = Union[
    NufftScattering, GaussianScattering
]  # , FourierSliceScattering]
from .specimen import ElectronCloud, ElectronGrid

Specimen = Union[ElectronCloud, ElectronGrid]
from .filters import (
    compute_anti_aliasing_filter,
    compute_whitening_filter,
    AntiAliasingFilter,
    WhiteningFilter,
)

Filter = Union[AntiAliasingFilter, WhiteningFilter]
from .mask import CircularMask, compute_circular_mask

Mask = CircularMask
from .ice import NullIce, ExponentialIce, EmpiricalIce

Ice = Union[NullIce, ExponentialIce, EmpiricalIce]
from .optics import compute_ctf_power, NullOptics, CTFOptics

Optics = Union[NullOptics, CTFOptics]
from .exposure import rescale_image, UniformExposure

Exposure = UniformExposure
from .detector import NullDetector, WhiteDetector

Detector = Union[NullDetector, WhiteDetector]
from .state import ParameterState
from .image import ScatteringImage, OpticsImage, DetectorImage, GaussianImage

Image = Union[ScatteringImage, OpticsImage, DetectorImage, GaussianImage]

del Union
