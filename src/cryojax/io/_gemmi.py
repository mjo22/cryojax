"""
Routines for converting Gemmi structures into voxel densities.
"""

import itertools

import numpy as np
from jaxtyping import Float, Int


def get_atom_info_from_gemmi_model(
    model,
) -> tuple[Float[np.ndarray, "N 3"], Int[np.ndarray, " N"]]:
    """
    Gets the atomic positions and element names from a Gemmi model.

    Parameters
    ----------

    model : Gemmi Class
        Gemmi model

    Returns
    -------
    atom_positions: numpy array
        Array of coordinates containing atomic positions
    atom_element_names: numpy array
        Array of atomic element names

    """
    atoms = extract_gemmi_atoms(model)
    atom_positions, atom_element_names = extract_atom_positions_and_names(atoms)
    return atom_positions, atom_element_names


def clean_gemmi_structure(structure=None):
    """
    Clean Gemmi Structure.

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


def extract_atom_positions_and_names(
    atoms,
) -> tuple[Float[np.ndarray, "N 3"], Int[np.ndarray, " N"]]:
    """
    Interpret Gemmi atoms and extract a single parameter type.

    Parameters
    ----------
    atoms : list (of list(s)) of Gemmi atoms
        Gemmi atom objects associated with each chain

    Returns
    -------
    positions : numpy array
        Array of atomic positions
    atomic_numbers : numpy array
        Array of atomic numbers

    TODO:
    - atomic charges?
    """
    # if list of Gemmi atoms, convert into a list of list
    if type(atoms[0]) != list:
        atoms = [atoms]

    positions = np.array([at.pos.tolist() for ch in atoms for at in ch])
    atomic_numbers = np.array([at.element.atomic_number for ch in atoms for at in ch])
    return positions, atomic_numbers
