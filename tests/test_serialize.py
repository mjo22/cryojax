import pytest

import numpy as np
from jax import config

import cryojax.simulator as cs

config.update("jax_enable_x64", True)


def test_deserialize_instrument(instrument):
    """Test Instrument deserialization"""
    assert (
        cs.Instrument.from_json(instrument.to_json()).to_json()
        == instrument.to_json()
    )


def test_deserialize_specimen(specimen):
    """Test Specimen deserialization"""
    assert (
        cs.Specimen.from_json(specimen.to_json()).to_json()
        == specimen.to_json()
    )


def test_deserialize_density(density):
    """Test specimen deserialization."""
    assert (
        cs.VoxelGrid.from_json(density.to_json()).to_json()
        == density.to_json()
    )


@pytest.mark.parametrize("filters_or_masks", ["filters", "masks"])
def test_deserialize_filters_and_masks(filters_or_masks, request):
    """Test filter and mask deserialization."""
    filter_or_mask = request.getfixturevalue(filters_or_masks)
    type = getattr(cs, filter_or_mask.__class__.__name__)
    test = type.from_json(filter_or_mask.to_json())
    assert test.to_json() == filter_or_mask.to_json()
    np.testing.assert_allclose(
        test.evaluate(), filter_or_mask.evaluate(), rtol=1e-6
    )


@pytest.mark.parametrize(
    "model", ["noisy_model", "maskless_model", "likelihood_model"]
)
def test_deserialize_model(model, request):
    """Test model deserialization."""
    model = request.getfixturevalue(model)
    cls = getattr(cs, model.__class__.__name__)
    test_model = cls.from_json(model.to_json())
    assert model.to_json() == test_model.to_json()
    np.testing.assert_allclose(test_model.render(), model.render(), rtol=1e-6)


def test_serialize_function():
    """Test function serialization as a dataclass field."""

    def f(x):
        return 2 * x

    x = 10
    kernel = cs.Custom(function=f)
    deserialized_kernel = cs.Custom.from_json(kernel.to_json())
    assert kernel(x) == deserialized_kernel(x)
