"""
Routines for interfacing with mdtraj
"""

from typing import Optional

import numpy as np
from jaxtyping import Float, Int


def get_atom_info_from_mdtraj(
    traj,
) -> tuple[Float[np.ndarray, "N 3"], Int[np.ndarray, " N"]]:
    """
    Gets the atomic information from an mdtraj trajectory.

    Parameters
    ----------
    traj : mdtraj trajectory

    Returns
    -------
    atom_positions : numpy array
        Atomic positions.
    atom_element_names : numpy arary
        Atomic number for each atom.

    """
    atom_element_names = [a.element.atomic_number for a in traj.top.atoms]
    atom_positions = traj.xyz
    return np.array(atom_positions), np.array(atom_element_names)


def mdtraj_load_from_file(
    path: str, top: Optional[str] = None
) -> tuple[Float[np.ndarray, "N 3"], Int[np.ndarray, " N"]]:
    """
    Loads a file using mdtraj and loads its atomic information
    """
    import mdtraj as md

    traj = md.load(path, top=top)
    return get_atom_info_from_mdtraj(traj)
