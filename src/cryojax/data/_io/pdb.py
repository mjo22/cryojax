"""
Read and write atomic models in various formats.
Large amounts of the code are adapted from the ioSPI package
"""

import numpy as np
from jaxtyping import Float, Int

from .gemmi import (
    clean_gemmi_structure,
    center_gemmi_model,
    extract_atom_positions_and_numbers,
    extract_gemmi_atoms,
)


def read_atoms_from_pdb(
    path: str,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
) -> tuple[Float[np.ndarray, "N 3"], Int[np.ndarray, " N"]]:
    """Read atomic information from a PDB file using Gemmi

    Parameters
    ----------
    path : string
        Path to PDB file.
    i_model : integer
        Optional, default: 0
        Index of the returned model in the Gemmi Structure.
    clean : bool
        Optional, default: True
        If True, use Gemmi remove_* methods to clean up structure.
    center : bool
        Optional, default: True
        If True, center the model so that its center of mass coincides with the origin.
    assemble: bool
        Optional, default: True
        If True, use Gemmi make_assembly to build biological object.

    Returns
    -------
    atom_positions: list of numpy arrays
        Array of coordinates containing atomic positions
    atom_element_numbers: array of ints
        Array of atomic element numbers

    Notes
    -----
    Currently Hydrogen atoms are not read in!
    We should look into adding hydrogens: does this slow things down
    appreciably?  Also, does it have a big effect on the scattering?
    """
    import gemmi

    structure = gemmi.read_structure(path)
    if clean:
        structure = clean_gemmi_structure(structure)
    if center:
        model = center_gemmi_model(structure[i_model])
    model = structure[i_model]
    if assemble:
        assembly = structure.assemblies[i_model]
        chain_naming = gemmi.HowToNameCopiedChain.AddNumber
        model = gemmi.make_assembly(assembly, model, chain_naming)

    atoms = extract_gemmi_atoms(model)
    atom_positions, atom_element_numbers = extract_atom_positions_and_numbers(atoms)
    return atom_positions, atom_element_numbers
