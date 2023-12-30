"""
Read and write atomic models in various formats.
Large amounts of the code are adapted from the ioSPI package
"""

__all__ = [
    "read_atomic_model_from_pdb",
]

from .load_atoms import clean_gemmi_structure


def read_atomic_model_from_pdb(path, i_model=0, clean=True, assemble=True):
    """Read Gemmi Model from PDB file.

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
    assemble: bool
        Optional, default: True
        If True, use Gemmi make_assembly to build biological object.

    Returns
    -------
    model : Gemmi Class
        Gemmi model

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
    return model
