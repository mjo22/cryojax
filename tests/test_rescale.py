from .test_pipeline import setup, optics_model

import pytest

import jax.numpy as jnp
from cryojax.utils import ifft


def test_normalization(optics_model):
    image = optics_model(dict(N=1.0, mu=0.0))
    assert pytest.approx(ifft(image).mean().item()) == 0.0
    assert pytest.approx(ifft(image).std().item()) == 1.0


def test_rescale(optics_model):
    N1, N2 = optics_model.scattering.shape
    mu, N = 0.5, 5.5
    image = optics_model(dict(N=N, mu=mu))
    assert pytest.approx(image[N1 // 2, N2 // 2].real.item()) == (N1 * N2) * mu
    assert pytest.approx(ifft(image).mean().item()) == mu
    assert (
        pytest.approx(
            jnp.linalg.norm(image.at[N1 // 2, N2 // 2].set(0.0)).item()
        )
        == (N1 * N2) * N
    )
    assert pytest.approx(ifft(image).std().item()) == N
