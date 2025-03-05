from jax import config
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.io import read_atoms_from_pdb

config.update("jax_enable_x64", True)


def test_read_single_pdb(sample_pdb_path):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        atom_filter="protein and not element H",
        is_assembly=False,
        i_model=None,
        get_b_factors=True,
    )

    assert atom_positions.ndim == 2
    assert atom_identities.shape == b_factors.shape
    assert atom_positions.shape[0] == atom_identities.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_single_pdb_no_b_factors(sample_pdb_path):
    atom_positions, atom_identities = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        atom_filter="protein and not element H",
        is_assembly=False,
        i_model=None,
        get_b_factors=False,
    )

    assert atom_positions.ndim == 2
    assert atom_positions.shape[0] == atom_identities.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_full_assembly_pdb(sample_pdb_path_assembly):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path_assembly,
        center=True,
        atom_filter="all",
        is_assembly=True,
        i_model=None,
        get_b_factors=True,
    )

    assert atom_positions.ndim == 2
    assert atom_identities.shape == b_factors.shape
    assert atom_positions.shape[0] == atom_identities.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 1380


def test_read_first_model_assembly_pdb(sample_pdb_path_assembly):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path_assembly,
        center=True,
        atom_filter="all",
        is_assembly=True,
        i_model=0,
        get_b_factors=True,
    )

    assert atom_positions.ndim == 2
    assert atom_identities.shape == b_factors.shape
    assert atom_positions.shape[0] == atom_identities.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 138
