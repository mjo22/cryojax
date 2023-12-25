from .load_atoms import clean_gemmi_structure


__all__ = [
    "read_atomic_model_from_cif",
]


def read_atomic_model_from_cif(path, i_model=0, clean=True, assemble=True):
    """Read Gemmi Model from CIF file.

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
    model : Gemmi Class
        Gemmi model
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
    return model
