"""
Image formation models, equipped with probabilistic models.
"""

from __future__ import annotations

__all__ = ["Distribution", "IndependentFourierGaussian"]

from abc import abstractmethod
from typing import Union, Optional, Any
from typing_extensions import override
from functools import cached_property

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .ice import NullIce, GaussianIce
from .detector import NullDetector, GaussianDetector
from .image import ImagePipeline, _PRNGKeyArrayLike
from ..utils import fftn
from ..typing import Real_, Image, RealImage
from ..core import Module, field


class Distribution(Module):
    """
    An imaging pipeline equipped with a probabilistic model.
    """

    pipeline: ImagePipeline = field()

    @abstractmethod
    def log_probability(self, observed: RealImage) -> Real_:
        """
        Evaluate the log-probability.

        Parameters
        ----------
        observed :
            The observed data in real space. This must be the same
            shape as ``scattering.shape``. Note that the user
            should preprocess the observed data before passing it
            to the image, such as applying the ``filters`` and
            ``masks``.
        """
        raise NotImplementedError

    def sample(
        self, key: Union[PRNGKeyArray, _PRNGKeyArrayLike], **kwargs: Any
    ):
        """
        Sample from the distribution.

        Parameters
        ----------
        key :
            The RNG key or key(s). See ``ImagePipeline.sample`` for
            documentation.
        """
        return self.pipeline.sample(key, **kwargs)

    def __call__(
        self,
        *,
        key: Optional[Union[PRNGKeyArray, _PRNGKeyArrayLike]] = None,
        observed: Optional[RealImage] = None,
        **kwargs: Any,
    ) -> Union[Image, Real_]:
        """
        If ``observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        if key is not None:
            return self.sample(key, **kwargs)
        elif observed is not None:
            return self.log_probability(observed, **kwargs)
        else:
            raise ValueError(
                "Must pass either a key or observed data as keywords to __call__ method"
            )


class IndependentFourierGaussian(Distribution):
    r"""
    A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.

    The variance is computed as

    .. math::
        Var[D(q)] + CTF(q)^2 Var[I(q)]

    where :math:`D` and :math:`I` are independent gaussian random variables in fourier
    space for the detector and ice, respectively.
    """

    def __check_init__(self):
        if not isinstance(
            self.pipeline.solvent, GaussianIce
        ) and not isinstance(
            self.pipeline.instrument.detector, GaussianDetector
        ):
            raise ValueError(
                "Either GaussianIce or GaussianDetector are required."
            )

    @override
    def log_probability(self, observed: RealImage) -> Real_:
        """Evaluate the log-likelihood of the gaussian noise model."""
        # Get residuals
        residuals = fftn(self.pipeline.render() - observed)
        # Compute loss
        loss = jnp.sum(
            (residuals * jnp.conjugate(residuals)) / (2 * self.variance)
        )
        # Divide by number of modes (parseval's theorem)
        loss = loss.real / residuals.size

        return loss

    @cached_property
    def variance(self) -> Union[Real_, RealImage]:
        pipeline = self.pipeline
        # Gather frequency coordinates
        freqs = pipeline.scattering.frequency_grid_in_angstroms
        # Variance from detector
        if not isinstance(pipeline.instrument.detector, NullDetector):
            variance = pipeline.instrument.detector.variance(freqs)
        else:
            variance = 0.0
        # Variance from ice
        if not isinstance(pipeline.solvent, NullIce):
            ctf = pipeline.instrument.optics(
                freqs, pose=pipeline.ensemble.pose
            )
            variance += ctf**2 * pipeline.solvent.variance(freqs)
        return variance
