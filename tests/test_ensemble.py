import pytest


def test_electron_density_indexing(density):
    stacked_density = density.from_list([density for _ in range(3)])
    assert stacked_density.n_stacked_dims == 1
    assert stacked_density[0].n_stacked_dims == 0
    assert stacked_density[:-1].n_stacked_dims == 1
