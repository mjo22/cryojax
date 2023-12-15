"""Read and write atomic models in various formats."""

import itertools
import os

import numpy as np


def _read_atomic_model_from_pdb(path, i_model=0, clean=True, assemble=True):
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


def _read_atomic_model_from_cif(path, i_model=0, clean=True, assemble=True):
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


def clean_gemmi_structure(structure=None):
    """Clean Gemmi Structure.

    Parameters
    ----------
    structure : Gemmi Class
        Gemmi Structure object

    Returns
    -------
    structure : Gemmi Class
        Same object, cleaned up of unnecessary atoms.

    """
    if structure is not None:
        structure.remove_alternative_conformations()
        structure.remove_hydrogens()
        structure.remove_waters()
        structure.remove_ligands_and_waters()
        structure.remove_empty_chains()

    return structure


def extract_gemmi_atoms(model, chains=None, split_chains=False):
    """
    Extract Gemmi atoms from the input Gemmi model, separated by chain.

    Parameters
    ----------
    model : Gemmi Class
        Gemmi model
    chains : list of strings
        chains to select, optional.
        If not provided, retrieve atoms from all chains.
    split_chains : bool
        Optional, default: False
        if True, keep the atoms from different chains in separate lists

    Returns
    -------
    atoms : list (or list of list(s)) of Gemmi atoms
        Gemmi atom objects, either concatenated or separated by chain
    """
    if chains is None:
        chains = [ch.name for ch in model]

    atoms = []
    for ch in model:
        if ch.name in chains:
            atoms.append([at for res in ch for at in res])

    if not split_chains:
        atoms = list(itertools.chain.from_iterable(atoms))

    return atoms


def extract_atomic_parameter(atoms, parameter_type, split_chains=False):
    """
    Interpret Gemmi atoms and extract a single parameter type.

    Parameters
    ----------
    atoms : list (of list(s)) of Gemmi atoms
        Gemmi atom objects associated with each chain
    parameter_type : string
        'cartesian_coordinates', 'form_factor_a', or 'form_factor_b'
    split_chains : bool
        Optional, default: False
        if True, keep the atoms from different chains in separate lists

    Returns
    -------
    atomic_parameter : list of floats, or list of lists of floats
        atomic parameter associated with each atom, optionally split by chain
    """
    # if list of Gemmi atoms, convert into a list of list
    if type(atoms[0]) != list:
        atoms = [atoms]

    if parameter_type == "cartesian_coordinates":
        atomic_parameter = [at.pos.tolist() for ch in atoms for at in ch]
    elif parameter_type == "electron_form_factor_a":
        atomic_parameter = [at.element.c4322.a for ch in atoms for at in ch]
    elif parameter_type == "electron_form_factor_b":
        atomic_parameter = [at.element.c4322.b for ch in atoms for at in ch]
    else:
        raise ValueError("Atomic parameter type not recognized.")

    # optionally preserve the list of lists (separated by chain) structure
    if split_chains:
        reshape = [0] + [len(ch) for ch in atoms]
        atomic_parameter = [
            atomic_parameter[reshape[i] : reshape[i] + reshape[i + 1]]
            for i in range(len(reshape) - 1)
        ]

    return atomic_parameter


def write_atomic_model(path, model=None):
    """Write Gemmi model to PDB or mmCIF file.

    Use Gemmi library to write an atomic model to file.

    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file.
    model : Gemmi Class
        Optional, default: gemmi.Model()
        Gemmi model

    Reference
    ---------
    See https://gemmi.readthedocs.io/en/latest/mol.html for a definition of
    gemmi objects.

    """
    import gemmi

    if model is None:
        model = gemmi.Model("model")
    is_pdb = path.lower().endswith(".pdb")
    is_cif = path.lower().endswith(".cif")
    if not (is_pdb or is_cif):
        raise ValueError("File format not recognized.")

    structure = gemmi.Structure()
    structure.add_model(model, pos=-1)
    structure.renumber_models()

    if is_cif:
        structure.make_mmcif_document().write_file(path)
    if is_pdb:
        structure.write_pdb(path)


def write_cartesian_coordinates(
    path, cartesian_coordinates_np=np.random.rand(10, 3)
):
    """Write Numpy array of cartesian coordinates to PDB or mmCIF file.

    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file
    cartesian_coordinates_np : numpy array
        Optional, default: np.random.rand(10,3)
        Second axis must be of dimension 3.

    -------

    """
    import gemmi

    is_pdb = path.lower().endswith(".pdb")
    is_cif = path.lower().endswith(".cif")
    if not (is_pdb or is_cif):
        raise ValueError("File format not recognized.")

    if cartesian_coordinates_np.shape[1] != 3:
        raise ValueError(
            "Numpy array of cartesian coordinates should be of shape (Natom, 3)."
        )

    structure = gemmi.Structure()
    structure.add_model(gemmi.Model("model"))
    structure.renumber_models()
    structure[0].add_chain("A")
    residue = gemmi.Residue()
    residue.name = "GLY"
    structure[0]["A"].add_residue(residue)

    for iat in np.arange(cartesian_coordinates_np.shape[0]):
        atom = gemmi.Atom()
        atom.pos = gemmi.Position(
            cartesian_coordinates_np[iat, 0],
            cartesian_coordinates_np[iat, 1],
            cartesian_coordinates_np[iat, 2],
        )
        atom.name = "CA"
        structure[0]["A"][0].add_atom(atom)

    if is_cif:
        structure.make_mmcif_document().write_file(path)
    if is_pdb:
        structure.write_pdb(path)
