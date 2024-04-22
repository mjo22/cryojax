from abc import abstractmethod
from typing import Optional

import equinox as eqx
from jaxtyping import Array, Complex, PRNGKeyArray

from .._instrument_config import InstrumentConfig


class AbstractScatteringTheory(eqx.Module, strict=True):
    """Base class for a scattering theory."""

    @abstractmethod
    def compute_fourier_squared_wavefunction_at_detector_plane(
        self,
        config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError
