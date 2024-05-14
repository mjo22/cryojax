"""
Read and write atomic models in various formats.
Large amounts of the code are adapted from the ioSPI package
"""

import pathlib
from typing import Literal, overload

import gemmi
import numpy as np
from jaxtyping import Float, Int

from ._gemmi import (
    center_gemmi_model,
    clean_gemmi_structure,
    extract_atom_b_factors,
    extract_atom_positions_and_numbers,
    extract_gemmi_atoms,
)


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
    get_b_factors: Literal[False] = False,
) -> tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
    get_b_factors: Literal[True] = True,
) -> tuple[
    Float[np.ndarray, "n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
    Float[np.ndarray, " n_atoms"],
]: ...


def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
    get_b_factors: bool = False,
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

    - `path`: Path to PDB file.
    - `i_model`: Index of the returned model in the `gemmi` Structure.
    - `clean`: If `True`, use `gemmi` remove_* methods to clean up structure.
    - `center`: If `True`, center the model so that its center of mass coincides
             with the origin.
    - `assemble`: If `True`, use `gemmi` make_assembly to build biological object.

    **Returns:**

    A tuple whose first element is a `numpy` array of coordinates containing
    atomic positions, and whose second element is an array of atomic element
    numbers. To be clear,

    ```python
    atom_positons, atom_element_numbers = read_atoms_from_pdb(...)
    ```

    Notes
    -----
    Currently Hydrogen atoms are not read in!
    We should look into adding hydrogens: does this slow things down
    appreciably?  Also, does it have a big effect on the scattering?
    """
    if pathlib.Path(filename).suffix != ".pdb":
        raise IOError(
            "Tried to read PDB file, but the filename does not have a .pdb "
            f"suffix. Got filename '{filename}'."
        )
    structure = gemmi.read_structure(str(filename))
    return _read_atom_info_from_structure(
        structure,
        i_model=i_model,
        clean=clean,
        center=center,
        assemble=assemble,
        get_b_factors=get_b_factors,
    )


@overload
def read_atoms_from_cif(
    filename: str | pathlib.Path,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
    get_b_factors: Literal[False] = False,
) -> tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_cif(
    filename: str | pathlib.Path,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
    get_b_factors: Literal[True] = True,
) -> tuple[
    Float[np.ndarray, "n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
    Float[np.ndarray, " n_atoms"],
]: ...


def read_atoms_from_cif(
    filename: str | pathlib.Path,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
    get_b_factors: bool = False,
) -> (
    tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]
    | tuple[
        Float[np.ndarray, "n_atoms 3"],
        Int[np.ndarray, " n_atoms"],
        Float[np.ndarray, " n_atoms"],
    ]
):
    """Read atomic information from an mmCIF file using `gemmi`.

    **Arguments:**

    - `path`: Path to mmCIF file.
    - `i_model`: Index of the returned model in the `gemmi` Structure.
    - `clean`: If `True`, use `gemmi` remove_* methods to clean up structure.
    - `center`: If `True`, center the model so that its center of mass coincides
             with the origin.
    - `assemble`: If `True`, use `gemmi` make_assembly to build biological object.

    **Returns:**

    A tuple whose first element is a `numpy` array of coordinates containing
    atomic positions, and whose second element is an array of atomic element
    numbers. To be clear,

    ```python
    atom_positons, atom_element_numbers = read_atoms_from_pdb(...)
    ```
    """
    if pathlib.Path(filename).suffix != ".cif":
        raise IOError(
            "Tried to read mmCIF file, but the filename does not have a .cif "
            f"suffix. Got filename '{filename}'."
        )
    cif_block = gemmi.cif.read(str(filename))[0]
    structure = gemmi.make_structure_from_block(cif_block)
    return _read_atom_info_from_structure(
        structure,
        i_model=i_model,
        clean=clean,
        center=center,
        assemble=assemble,
        get_b_factors=get_b_factors,
    )


def _read_atom_info_from_structure(
    structure,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
    get_b_factors: bool = False,
):
    atoms = _read_atoms_from_structure(
        structure, i_model=i_model, clean=clean, center=center, assemble=assemble
    )
    atom_positions, atom_element_numbers = extract_atom_positions_and_numbers(atoms)

    if not get_b_factors:
        return atom_positions, atom_element_numbers
    else:
        atom_b_factors = extract_atom_b_factors(atoms)
        return atom_positions, atom_element_numbers, atom_b_factors


def _read_atoms_from_structure(
    structure,
    i_model: int = 0,
    clean: bool = True,
    center: bool = True,
    assemble: bool = True,
):
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

    return atoms
