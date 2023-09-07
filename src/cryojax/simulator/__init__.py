__all__ = [
    # Functional API
    "rotate_rpy",
    "rotate_wxyz",
    "shift_phase",
    "make_euler_rotation",
    "project_with_nufft",
    "extract_slice",
    "compute_lowpass_filter",
    "compute_whitening_filter",
    "compute_circular_mask",
    "compute_ctf",
    "rescale_image",
    "measure_image",
    # Image configuration
    "ImageConfig",
    "NufftScattering",
    "FourierSliceScattering"
    # Specimen representations
    "ElectronCloud",
    "ElectronGrid",
    # Filters
    "LowpassFilter",
    "WhiteningFilter",
    # Masks
    "CircularMask",
    # Model parameter configuration
    "PipelineState",
    ## Pose
    "EulerPose",
    "QuaternionPose",
    ## Ice
    "EmpiricalIce",
    "ExpIce",
    ## Optics
    "CTFOptics",
    ## Electron exposure models
    "UniformExposure",
    ## Detector models
    "CountingDetector",
    "WhiteDetector",
    # Image models
    "ScatteringImage",
    "OpticsImage",
    "DetectorImage",
    "GaussianImage",
    # Kernels
    "Sum",
    "Product",
    "Constant",
    "Exp",
    "Gaussian",
    "Empirical",
    "Custom",
    # Type hints
    "ScatteringConfig",
    "Pose",
    "Filter",
    "Mask",
    "Optics",
    "Exposure",
    "Ice",
    "Detector",
    "Image",
    "Kernel",
]

from typing import Union

from .kernel import Sum, Product, Constant, Exp, Gaussian, Empirical, Custom

Kernel = Union[Sum, Product, Constant, Exp, Gaussian, Empirical, Custom]

from .pose import (
    rotate_rpy,
    rotate_wxyz,
    shift_phase,
    make_euler_rotation,
    EulerPose,
    QuaternionPose,
)

Pose = Union[EulerPose, QuaternionPose]
"""Type alias for Pose subclasses"""

from .scattering import (
    project_with_nufft,
    extract_slice,
    ImageConfig,
    NufftScattering,
    FourierSliceScattering,
)

ScatteringConfig = Union[NufftScattering, FourierSliceScattering]
"""Type alias for ScatteringConfig subclasses"""

from .specimen import ElectronCloud, ElectronGrid

Specimen = Union[ElectronCloud, ElectronGrid]
"""Type alias for Specimen subclasses"""

from .filters import (
    compute_lowpass_filter,
    compute_whitening_filter,
    LowpassFilter,
    WhiteningFilter,
)

Filter = Union[LowpassFilter, WhiteningFilter]
"""Type alias for Filter subclasses"""

from .masks import CircularMask, compute_circular_mask

Mask = CircularMask
"""Type alias for Mask subclasses"""

from .ice import NullIce, EmpiricalIce, ExpIce

Ice = Union[NullIce, ExpIce, EmpiricalIce]
"""Type alias for Ice subclasses"""

from .optics import compute_ctf, NullOptics, CTFOptics

Optics = Union[NullOptics, CTFOptics]
"""Type alias for Optics subclasses"""

from .exposure import rescale_image, NullExposure, UniformExposure

Exposure = Union[NullExposure, UniformExposure]
"""Type alias for Exposure subclasses"""

from .detector import (
    measure_image,
    NullDetector,
    CountingDetector,
    WhiteDetector,
)

Detector = Union[NullDetector, CountingDetector, WhiteDetector]
"""Type alias for Detector subclasses"""

from .state import PipelineState
from .image import ScatteringImage, OpticsImage, DetectorImage
from .likelihood import GaussianImage

Image = Union[ScatteringImage, OpticsImage, DetectorImage, GaussianImage]
"""Type alias for Image subclasses"""
