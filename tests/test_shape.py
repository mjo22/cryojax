import pytest


@pytest.mark.parametrize("model", ["noisy_model", "filtered_and_masked_model"])
def test_shape(model, request):
    """Make sure shapes are as expected"""
    model = request.getfixturevalue(model)
    image = model()
    padded_image = model(view=False)
    assert image.shape == model.scattering.manager.shape
    assert padded_image.shape == model.scattering.manager.padded_shape
