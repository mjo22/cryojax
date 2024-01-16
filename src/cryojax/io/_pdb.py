"""
Read and write atomic models in various formats.
Large amounts of the code are adapted from the ioSPI package
"""

__all__ = [
    "read_atoms_from_pdb",
]

from ._gemmi import (
    clean_gemmi_structure,
    extract_gemmi_atoms,
    extract_atom_positions_and_names,
)


def read_atoms_from_pdb(path, i_model=0, clean=True, assemble=True):
    """Read atomic information from a PDB file using Gemmi

    Parameters
    ----------At
    path : string
        Path to PDB file.
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

    Notes
    -----
    Currently Hydrogen atoms are not read in!
    We should look into adding hydrogens, potentially.
    """
    import gemmi

    structure = gemmi.read_structure(path)
    if clean:
        structure = clean_gemmi_structure(structure)
    model = structure[i_model]
    if assemble:
        assembly = structure.assemblies[i_model]
        chain_naming = gemmi.HowToNameCopiedChain.AddNumber
        model = gemmi.make_assembly(assembly, model, chain_naming)

    atoms = extract_gemmi_atoms(model)
    atom_positions, atom_element_names = extract_atom_positions_and_names(
        atoms
    )
    return atom_positions, atom_element_names
