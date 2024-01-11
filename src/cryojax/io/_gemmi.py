"""
Routines for converting Gemmi structures into voxel densities.
"""
__all__ = [
    "get_scattering_info_from_gemmi_model",
    "clean_gemmi_structure",
    "extract_gemmi_atoms",
]

import numpy as np
import itertools

from cryojax.io.load_atoms import (
    get_form_factor_params,
)


def get_scattering_info_from_gemmi_model(model):
    atoms = extract_gemmi_atoms(model)
    atom_positions, atom_element_names = extract_atom_positions_and_names(
        atoms
    )

    a_vals, b_vals = get_form_factor_params(atom_element_names)
    return np.array(atom_positions), np.array(a_vals), np.array(b_vals)


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


def extract_atom_positions_and_names(atoms):
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

    positions = [at.pos.tolist() for ch in atoms for at in ch]
    names = [at.element.name for ch in atoms for at in ch]
    return positions, names
