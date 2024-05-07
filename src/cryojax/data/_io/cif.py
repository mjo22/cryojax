import numpy as np
from jaxtyping import Float, Int

from .gemmi import (
    clean_gemmi_structure,
    extract_atom_positions_and_names,
    extract_gemmi_atoms,
)


def read_atoms_from_cif(
    path, i_model=0, clean=True, assemble=True
) -> tuple[Float[np.ndarray, "N 3"], Int[np.ndarray, " N"]]:
    """Read atomic positions and element names from a mmCIF file using Gemmi

    Parameters
    ----------
    path : string
        Path to mmCIF file.
    i_model : integer
        Optional, default: 0
        Index of the returned model in the Gemmi Structure.
    clean : bool
        Optional, default: True
        If True, use Gemmi remove_* methods to clean up structure.
    assemble: bool
        Optional, default: True
        If True, use Gemmi make_assembly to build biological object.

    Returns
    -------
    atom_positions: list of numpy arrays
        List of coordinates containing atomic positions
    atom_element_names: list of strings
        List of atomic element names
    """
    import gemmi

    cif_block = gemmi.cif.read(path)[0]
    structure = gemmi.make_structure_from_block(cif_block)
    if clean:
        structure = clean_gemmi_structure(structure)
    model = structure[i_model]
    if assemble:
        assembly = structure.assemblies[i_model]
        chain_naming = gemmi.HowToNameCopiedChain.AddNumber
        model = gemmi.make_assembly(assembly, model, chain_naming)
    atoms = extract_gemmi_atoms(model)
    atom_positions, atom_element_names = extract_atom_positions_and_names(atoms)
    return atom_positions, atom_element_names
