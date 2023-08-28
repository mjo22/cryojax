import pytest


@pytest.mark.parametrize(
    "model", ["scattering_model", "optics_model", "noisy_model"]
)
def test_shape(model, request):
    """Make sure shapes are as expected"""
    model = request.getfixturevalue(model)
    image = model()
    padded_image = model(view=False)
    assert image.shape == model.scattering.shape
    assert padded_image.shape == model.scattering.padded_shape
