from abc import abstractmethod
from typing import Optional

import equinox as eqx
from jaxtyping import Array, Complex, PRNGKeyArray

from .._config import ImageConfig
from .._instrument import Instrument


class AbstractScatteringTheory(eqx.Module, strict=True):

    @abstractmethod
    def compute_fourier_squared_wavefunction_at_detector_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError
