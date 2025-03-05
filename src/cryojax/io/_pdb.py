"""
Read atomic information from a PDB file using functions and objects adapted from `mdtraj`.
"""

import os
import pathlib
from typing import Literal, Optional, overload

import mdtraj
import numpy as np
from jaxtyping import Float, Int
from mdtraj.utils import in_units_of

from ._mdtraj import PDBReader


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    center: bool = False,
    atom_filter: str = "all",
    is_assembly: bool = False,
    i_model: Optional[int] = None,
    get_b_factors: Literal[False] = False,
    *,
    standard_names: bool = True,
    top: Optional[mdtraj.Topology] = None,
) -> tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    center: bool = False,
    atom_filter: str = "all",
    is_assembly: bool = False,
    i_model: Optional[int] = None,
    get_b_factors: Literal[False] = False,
    *,
    standard_names: bool = True,
    top: Optional[mdtraj.Topology] = None,
) -> tuple[
    Float[np.ndarray, "n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
    Float[np.ndarray, " n_atoms"],
]: ...


def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    center: bool = False,
    atom_filter: str = "all",
    is_assembly: bool = False,
    i_model: Optional[int] = None,
    get_b_factors: Literal[False] = False,
    *,
    standard_names: bool = True,
    top: Optional[mdtraj.core.topology.Topology] = None,
) -> (
    tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]
    | tuple[
        Float[np.ndarray, "n_atoms 3"],
        Int[np.ndarray, " n_atoms"],
        Float[np.ndarray, " n_atoms"],
    ]
):
    """Read atomic information from a PDB file using `gemmi`.

    **Arguments:**

    - `filename`: Path to PDB file.
    - `center`: If `True`, center the model so that its center of mass coincides
             with the origin.
    - `filter`: A selection string in `mdtraj`'s format.
    - `is_assembly`: If the pdb file contains multiple models, set this to `True`.
            Warning: if your pdb is a trajectory, all frames will be loaded.
    - `i_model`: Index of the returned mode.
        Should only be used if `is_assembly` is `True`.
    - `get_b_factors`: If `True`, return the B-factors of the atoms.
    - `standard_names` : bool, default=True
        If True, non-standard atomnames and residuenames are standardized to conform
        with the current PDB format version. If set to false, this step is skipped.
    - `top` : mdtraj.core.Topology, default=None
        if you give a topology as input the topology won't be parsed from the pdb file
        it saves time if you have to parse a big number of files

    **Returns:**

    A tuple whose first element is a `numpy` array of coordinates containing
    atomic positions, and whose second element is an array of atomic element
    numbers. To be clear,

    ```python
    atom_positons, atom_element_numbers = read_atoms_from_pdb(...)
    ```

    """
    from mdtraj import Trajectory

    if not isinstance(filename, (str, os.PathLike)):
        raise TypeError(
            "filename must be of type string or path-like for load_pdb. "
            "you supplied %s" % type(filename),
        )

    with PDBReader(
        filename,
        is_assembly=is_assembly,
        i_model=i_model,
        standard_names=standard_names,
        top=top,
    ) as f:
        topology = f.topology
        atom_indices = topology.select(atom_filter)
        topology = topology.subset(atom_indices)

        if f.unitcell_angles is not None and f.unitcell_lengths is not None:
            unitcell_lengths = np.array([f.unitcell_lengths])
            unitcell_angles = np.array([f.unitcell_angles])
        else:
            unitcell_lengths = None
            unitcell_angles = None

        coords = f.positions[np.newaxis, atom_indices, ...]
        assert coords.ndim == 3, "internal shape error"
        b_factors = f.bfactors[atom_indices]
        atom_identities = f.atomic_numbers[atom_indices]

        in_units_of(coords, f.distance_unit, Trajectory._distance_unit, inplace=True)
        in_units_of(
            unitcell_lengths,
            f.distance_unit,
            Trajectory._distance_unit,
            inplace=True,
        )

    time = np.arange(len(coords))

    traj = Trajectory(
        xyz=coords,
        time=time,
        topology=topology,
        unitcell_lengths=unitcell_lengths,
        unitcell_angles=unitcell_angles,
    )

    if center:
        traj.center_coordinates()

    atom_positions = traj.xyz[0] * 10.0  # nm -> angstroms

    if get_b_factors:
        return atom_positions, atom_identities, b_factors
    else:
        return atom_positions, atom_identities
