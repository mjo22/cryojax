"""
Read atomic information from a PDB file using functions and objects adapted from `mdtraj`.
"""

import dataclasses
import gzip
import pathlib
from io import StringIO
from typing import Literal, Optional, overload
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative
from urllib.request import urlopen

import mdtraj
import numpy as np
from jaxtyping import Float, Int
from mdtraj.core.topology import Topology
from mdtraj.formats.pdb.pdbfile import PDBTrajectoryFile
from mdtraj.formats.pdb.pdbstructure import PdbStructure
from mdtraj.utils import in_units_of, open_maybe_zipped


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


@overload
def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    center: bool = False,
    get_b_factors: Literal[False] = False,
    *,
    atom_filter: str = "all",
    standard_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
) -> tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    center: bool = False,
    get_b_factors: Literal[True] = True,
    *,
    atom_filter: str = "all",
    standard_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
) -> tuple[
    Float[np.ndarray, "n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
    Float[np.ndarray, " n_atoms"],
]: ...


@overload
def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    center: bool = False,
    get_b_factors: bool = False,
    *,
    atom_filter: str,
    standard_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
) -> (
    tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]
    | tuple[
        Float[np.ndarray, "n_atoms 3"],
        Int[np.ndarray, " n_atoms"],
        Float[np.ndarray, " n_atoms"],
    ]
): ...


def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    center: bool = False,
    get_b_factors: bool = False,
    *,
    atom_filter: str = "all",
    standard_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
) -> (
    tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]
    | tuple[
        Float[np.ndarray, "n_atoms 3"],
        Int[np.ndarray, " n_atoms"],
        Float[np.ndarray, " n_atoms"],
    ]
):
    """Read atomic information from a PDB file. This object
    wraps the `cryojax.io.PDBReader` class into a function
    interface to accomodate most use cases in cryo-EM.

    **Arguments:**

    - `filename_or_url`:
        The name of the PDB/mmCIF file to open. Can be a URL.
    - `center`:
        If `True`, center the model so that its center of mass coincides
        with the origin.
    - `get_b_factors`:
        If `True`, return the B-factors of the atoms.
    - `atom_filter`:
        A selection string in `mdtraj`'s format. See `mdtraj` for documentation.
    - `standard_names`:
        If `True`, non-standard atomnames and residuenames are standardized to conform
        with the current PDB format version. If set to `False`, this step is skipped.
    - `topology`:
        If you give a topology as input, the topology won't be parsed from the pdb file
        it saves time if you have to parse a big number of files

    **Returns:**

    A tuple whose first element is a `numpy` array of coordinates containing
    atomic positions, and whose second element is an array of atomic element
    numbers. To be clear,

    ```python
    atom_positons, atom_element_numbers = read_atoms_from_pdb(...)
    ```

    !!! warning

        If your pdb is a trajectory, all frames will be loaded.
        In particular, the output arrays `atom_positions` and
        `atom_element_numbers` will contain all atoms from all
        trajectories.
    """

    with AtomicModelReader(filename_or_url, standard_names, topology) as pdb_reader:
        topology = pdb_reader.topology
        atom_indices = topology.select(atom_filter)
        topology = topology.subset(atom_indices)

        is_unitcell_read = (
            pdb_reader.unitcell_angles is not None
            and pdb_reader.unitcell_lengths is not None
        )
        if is_unitcell_read:
            unitcell_lengths = np.array([pdb_reader.unitcell_lengths])
            unitcell_angles = np.array([pdb_reader.unitcell_angles])
        else:
            unitcell_lengths = None
            unitcell_angles = None

        raw_atom_positions = pdb_reader.atom_positions[np.newaxis, atom_indices, ...]
        assert raw_atom_positions.ndim == 3, "internal shape error"
        b_factors = pdb_reader.b_factors[atom_indices]
        atom_identities = pdb_reader.atomic_numbers[atom_indices]

        in_units_of(
            raw_atom_positions,
            "angstroms",
            mdtraj.Trajectory._distance_unit,
            inplace=True,
        )
        in_units_of(
            unitcell_lengths, "angstroms", mdtraj.Trajectory._distance_unit, inplace=True
        )

    time = np.arange(len(raw_atom_positions))

    traj = mdtraj.Trajectory(
        xyz=raw_atom_positions,
        time=time,
        topology=topology,
        unitcell_lengths=unitcell_lengths,
        unitcell_angles=unitcell_angles,
    )

    if center:
        traj.center_coordinates()

    # Convert nm -> angstroms
    atom_positions = traj.xyz[0] * 10.0  # type: ignore

    if get_b_factors:
        return atom_positions, atom_identities, b_factors
    else:
        return atom_positions, atom_identities


@dataclasses.dataclass
class AtomicModelReader:
    """A PDB file loader that loads the necessary information for
    cryo-EM. This object wraps and is based on the `PDBTrajectoryFile`
    from `mdtraj`.

    **Attributes:**

    - `atom_positions`: The cartesian coordinates of all of the atoms in each frame.
    - `atomic_numbers`: The atomic numbers of all of the atoms in each frame.
    - `b_factors`: The B-factors of all of the atoms in each frame.
    - `topology`: mdtraj.core.Topology, default=None
        if you give a topology as input the topology won't be parsed from the pdb file
        it saves time if you have to parse a big number of files
    - `unitcell_lengths`: The unitcell lengths (3-tuple) in this PDB file. May be None
    - `unitcell_angles`: The unitcell angles (3-tuple) in this PDB file. May be None
    """

    atom_positions: np.ndarray
    atomic_numbers: np.ndarray
    b_factors: np.ndarray
    topology: Topology
    unitcell_lengths: tuple[float, float, float] | None
    unitcell_angles: tuple[float, float, float] | None

    def __init__(
        self,
        filename_or_url: str | pathlib.Path,
        standard_names: bool = True,
        topology: Optional[Topology] = None,
    ):
        """**Arguments:**

        - `filename_or_url`: The name of the PDB/mmCIF file to open. Can be a URL.
        - `standard_names` : bool, default=True
            If `True`, non-standard atomnames and residuenames are standardized to conform
            with the current PDB format version. If set to false, this step is skipped.
        - `topology` : mdtraj.core.Topology, default=None
            If the `topology` is passed as input, it won't be parsed from the PDB file.
            This saves time if you have to parse a big number of files.
        """
        # Set state of the loader
        self._is_open = True
        # Check for errors
        filename_or_url = str(filename_or_url)
        if not (".pdb" in filename_or_url or ".pdb.gz" in filename_or_url):
            raise ValueError(
                "File format not recognized. Currently, supported file "
                "formats are '.pdb' and '.pdb.gz'."
            )

        # Setup I/O
        if _is_url(filename_or_url):
            filename_or_url = str(filename_or_url)
            self._file = urlopen(filename_or_url)
            if filename_or_url.lower().endswith(".gz"):
                self._file = gzip.GzipFile(fileobj=self._file)
            self._file = StringIO(self._file.read().decode("utf-8"))

        else:
            filename_or_url = pathlib.Path(filename_or_url)
            _validate_pdb_file(filename_or_url)
            self._file = open_maybe_zipped(filename_or_url, "r")

        # Load properties into the object
        properties_dict = _load_pdb_reader_properties_dict(
            self._file, topology, standard_names
        )
        for k, v in properties_dict.items():
            setattr(self, k, v)

    @property
    def is_closed(self):
        return not self._is_open

    def close(self):
        """Close the PDB file"""
        if self._is_open:
            if hasattr(self, "_file"):
                self._file.close()
        self._is_open = False

    def __del__(self):
        self.close()
        del self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _load_pdb_reader_properties_dict(
    file,
    topology: Optional[Topology],
    standard_names,
):
    pdb = PdbStructure(file, load_all_models=True)

    positions = []
    bfactors = []
    atomic_numbers = []

    # load all of the positions (from every model)
    for i, model in enumerate(pdb.iter_models(use_all_models=True)):
        for chain in model.iter_chains():
            for residue in chain.iter_residues():
                for atom in residue.atoms:
                    positions.append(atom.get_position())
                    bfactors.append(atom.get_temperature_factor())
                    atomic_numbers.append(atom.element.atomic_number)

    atom_positions = np.array(positions)
    b_factors = np.array(bfactors)
    atomic_numbers = np.array(atomic_numbers)

    ## The atom positions read from the PDB file
    unitcell_lengths = pdb.get_unit_cell_lengths()
    unitcell_angles = pdb.get_unit_cell_angles()

    # Load the topology if None is given
    if topology is None:
        topology = _make_topology(pdb, positions, standard_names)

    return dict(
        atom_positions=atom_positions,
        b_factors=b_factors,
        atomic_numbers=atomic_numbers,
        unitcell_lengths=unitcell_lengths,
        unitcell_angles=unitcell_angles,
        topology=topology,
    )


def _make_topology(pdb, atom_positions, standard_names):
    topology = Topology()

    atomByNumber = {}
    for i, model in enumerate(pdb.iter_models(use_all_models=True)):
        for chain in model.iter_chains():
            c = topology.add_chain(chain.chain_id)
            for residue in chain.iter_residues():
                resName = residue.get_name()
                if (
                    resName in PDBTrajectoryFile._residueNameReplacements
                    and standard_names
                ):
                    resName = PDBTrajectoryFile._residueNameReplacements[resName]
                r = topology.add_residue(resName, c, residue.number, residue.segment_id)
                if resName in PDBTrajectoryFile._atomNameReplacements and standard_names:
                    atomReplacements = PDBTrajectoryFile._atomNameReplacements[resName]
                else:
                    atomReplacements = {}
                for atom in residue.atoms:
                    atomName = atom.get_name()
                    if atomName in atomReplacements:
                        atomName = atomReplacements[atomName]
                    atomName = atomName.strip()
                    elem = atom.element
                    if elem is None:
                        elem = PDBTrajectoryFile._guess_element(
                            atomName, residue.name, len(residue)
                        )

                    newAtom = topology.add_atom(
                        atomName,
                        elem,
                        r,
                        serial=atom.serial_number,
                        formal_charge=atom.formal_charge,
                    )
                    atomByNumber[atom.serial_number] = newAtom

    topology.create_standard_bonds()
    topology.create_disulfide_bonds(atom_positions)

    """
    This might not be necessary

    # Add bonds based on CONECT records.
    connectBonds = []
    for l, model in enumerate(biopython_struct.get_models()):
        if i_model is not None and l != i_model:
            continue
        for connect in model.connects:
            i = connect[0]
            for j in connect[1:]:
                if i in atomByNumber and j in atomByNumber:
                    connectBonds.append((atomByNumber[i], atomByNumber[j]))
    if len(connectBonds) > 0:
        # Only add bonds that don't already exist.
        existingBonds = {(bond.atom1, bond.atom2) for bond in topology.bonds}
        for bond in connectBonds:
            if bond not in existingBonds and (bond[1], bond[0]) not in existingBonds:
                topology.add_bond(bond[0], bond[1])
                existingBonds.add(bond)
    """

    return topology


def _validate_pdb_file(filename):
    if filename.suffixes not in [[".pdb"], [".pdb", ".gz"], [".cif"]]:
        raise ValueError(
            "PDB filename must have suffix `.pdb`, `.pdb.gz`, or 'cif'. "
            f"Got filename {filename}."
        )


def _is_url(url):
    """Check to see if a URL has a valid protocol.
    from pandas/io.common.py Copyright 2014 Pandas Developers
    Used under the BSD licence
    """
    try:
        return urlparse(url).scheme in _VALID_URLS
    except (AttributeError, TypeError):
        return False
