"""
Read atomic information from a PDB file using functions and objects adapted from `mdtraj`.
"""

import gzip
import pathlib
from io import StringIO
from typing import Literal, NamedTuple, Optional, overload
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative
from urllib.request import urlopen

import mdtraj
import numpy as np
from jaxtyping import Float, Int
from mdtraj.core import element as elem
from mdtraj.core.topology import Topology
from mdtraj.formats.pdb.pdbfile import PDBTrajectoryFile
from mdtraj.formats.pdb.pdbstructure import PdbStructure
from mdtraj.utils import open_maybe_zipped


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


@overload
def read_atoms_from_pdb(  # type: ignore
    filename_or_url: str | pathlib.Path,
    center: bool = False,
    loads_b_factors: Literal[False] = False,
    *,
    select: str = "all",
    standardizes_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
) -> tuple[Float[np.ndarray, "n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    center: bool = False,
    loads_b_factors: Literal[True] = True,
    *,
    select: str = "all",
    standardizes_names: bool = True,
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
    loads_b_factors: bool = False,
    *,
    select: str,
    standardizes_names: bool = True,
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
    loads_b_factors: bool = False,
    *,
    select: str = "all",
    standardizes_names: bool = True,
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
    wraps the `cryojax.io.AtomicModelReader` class into a function
    interface to accomodate most use cases in cryo-EM.

    **Arguments:**

    - `filename_or_url`:
        The name of the PDB/mmCIF file to open. Can be a URL.
    - `center`:
        If `True`, center the model so that its center of mass coincides
        with the origin.
    - `loads_b_factors`:
        If `True`, return the B-factors of the atoms.
    - `select`:
        A selection string in `mdtraj`'s format. See `mdtraj` for documentation.
    - `standardizes_names`:
        If `True`, non-standard atom names and residue names are standardized to conform
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
    with AtomicModelFile(filename_or_url) as pdb_file:
        # Read file
        info, topology = pdb_file.read(
            standardizes_names=standardizes_names,
            topology=topology,
        )
    # ... get indices from filter
    atom_indices = topology.select(select)
    # ... get filtered attributes
    atom_positions = info.atom_positions[atom_indices, ...]
    atom_identities = info.atom_identities[atom_indices]
    if center:
        atom_masses = info.atom_masses[atom_indices]
        atom_positions = _center_atom_coordinates(atom_positions, atom_masses)

    if loads_b_factors:
        b_factors = info.b_factors[atom_indices]
        return atom_positions, atom_identities, b_factors
    else:
        return atom_positions, atom_identities


def _center_atom_coordinates(atom_positions, atom_masses):
    com_position = atom_positions.astype("float64").T.dot(atom_masses / atom_masses.sum())
    return atom_positions - com_position


class AtomInfo(NamedTuple):
    """A struct for the info of individual atoms.

    **Attributes:**

    - `atom_positions`: The cartesian coordinates of all of the atoms in each frame.
    - `atom_identities`: The atomic numbers of all of the atoms in each frame.
    - `atom_masses`: The mass of each atom.
    - `b_factors`: The B-factors of all of the atoms in each frame.
    """

    atom_positions: np.ndarray
    atom_identities: np.ndarray
    atom_masses: np.ndarray
    b_factors: np.ndarray


class AtomicModelFile:
    """A PDB file loader that loads the necessary information for
    cryo-EM. This object is based on the `PDBTrajectoryFile`
    from `mdtraj`.
    """

    def __init__(self, filename_or_url: str | pathlib.Path):
        """**Arguments:**

        - `filename_or_url`:
            The name of the PDB file to open. Can be a URL.
        """
        # Set field that says we are reading the file
        self._is_open = True
        # Setup I/O
        if _is_url(filename_or_url):
            filename_or_url = str(filename_or_url)
            file_obj = urlopen(filename_or_url)
            if filename_or_url.lower().endswith(".gz"):
                file_obj = gzip.GzipFile(fileobj=file_obj)
            self._file = StringIO(file_obj.read().decode("utf-8"))

        else:
            filename_or_url = pathlib.Path(filename_or_url)
            _validate_pdb_file(filename_or_url)
            self._file = open_maybe_zipped(filename_or_url, "r")
        # Create the PDB structure via `mdtraj`
        self._pdb_structure = PdbStructure(self._file, load_all_models=True)

    @property
    def pdb_structure(self) -> PdbStructure:
        return self._pdb_structure

    def read(
        self,
        *,
        standardizes_names: bool = True,
        topology: Optional[Topology] = None,
    ) -> tuple[AtomInfo, Topology]:
        """Load properties from the PDB reader.

        **Arguments:**

        - `standardizes_names`:
            If `True`, non-standard atom names and residue names are standardized to
            conform with the current PDB format version. If `False`, this step is skipped.
        - `topology`:
            If the `topology` is passed as input, it won't be parsed from the PDB file.
            This saves time if you have to parse a big number of files.

        **Returns:**

        A tuple of the `AtomInfo` dataclass and the `mdtraj.Topology`.
        """
        atom_info, topology = _load_atom_info(
            self._pdb_structure,
            topology,
            standardizes_names,
        )

        return atom_info, topology

    @property
    def is_open(self):
        return self._is_open

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


def _make_topology(
    pdb: PdbStructure,
    atom_positions: np.ndarray | list,
    standardizes_names: bool = True,
) -> Topology:
    topology = Topology()
    if standardizes_names:
        residue_name_replacements, atom_name_replacements = (
            _load_name_replacement_tables()
        )
    else:
        residue_name_replacements, atom_name_replacements = {}, {}
    atom_by_number = {}
    for model in pdb.iter_models(use_all_models=True):
        for chain in model.iter_chains():
            c = topology.add_chain(chain.chain_id)
            for residue in chain.iter_residues():
                residue_name = residue.get_name()
                if residue_name in residue_name_replacements and standardizes_names:
                    residue_name = residue_name_replacements[residue_name]
                r = topology.add_residue(
                    residue_name, c, residue.number, residue.segment_id
                )
                if residue_name in atom_name_replacements and standardizes_names:
                    atom_replacements = atom_name_replacements[residue_name]
                else:
                    atom_replacements = {}
                for atom in residue.atoms:
                    atom_name = atom.get_name()
                    if atom_name in atom_replacements:
                        atom_name = atom_replacements[atom_name]
                    atom_name = atom_name.strip()
                    element = atom.element
                    if element is None:
                        element = _guess_element(atom_name, residue.name, len(residue))

                    new_atom = topology.add_atom(
                        atom_name,
                        element,
                        r,
                        serial=atom.serial_number,
                        formal_charge=atom.formal_charge,
                    )
                    atom_by_number[atom.serial_number] = new_atom

    topology.create_standard_bonds()
    topology.create_disulfide_bonds(atom_positions)

    return topology


def _load_atom_info(
    pdb: PdbStructure,
    topology: Optional[Topology],
    standardizes_names: bool,
):
    atom_positions, b_factors, atom_identities, atom_masses = [], [], [], []
    # load all of the positions (from every model)
    for model in pdb.iter_models(use_all_models=True):
        for chain in model.iter_chains():
            for residue in chain.iter_residues():
                for atom in residue.atoms:
                    # ... make sure this is read in angstroms?
                    atom_positions.append(atom.get_position())
                    atom_identities.append(atom.element.atomic_number)
                    atom_masses.append(atom.element.mass)
                    b_factors.append(atom.get_temperature_factor())

    # Load the topology if None is given
    if topology is None:
        topology = _make_topology(
            pdb,
            atom_positions,
            standardizes_names,
        )

    # Gather properties and return
    atom_positions = np.array(atom_positions)
    b_factors = np.array(b_factors)
    atom_identities = np.array(atom_identities)
    atom_masses = np.array(atom_masses)

    atom_info = AtomInfo(
        atom_positions=atom_positions,
        atom_identities=atom_identities,
        atom_masses=atom_masses,
        b_factors=b_factors,
    )

    return atom_info, topology


def _guess_element(atom_name, residue_name, residue_length):
    "Try to guess the element name. Based on `mdtraj.PDBTrajectoryFile._guess_element`."
    upper = atom_name.upper()
    if upper.startswith("CL"):
        element = elem.chlorine
    elif upper.startswith("NA"):
        element = elem.sodium
    elif upper.startswith("MG"):
        element = elem.magnesium
    elif upper.startswith("BE"):
        element = elem.beryllium
    elif upper.startswith("LI"):
        element = elem.lithium
    elif upper.startswith("K"):
        element = elem.potassium
    elif upper.startswith("ZN"):
        element = elem.zinc
    elif residue_length == 1 and upper.startswith("CA"):
        element = elem.calcium

    # TJL has edited this. There are a few issues here. First,
    # parsing for the element is non-trivial, so I do my best
    # below. Second, there is additional parsing code in
    # pdbstructure.py, and I am unsure why it doesn't get used
    # here...
    elif residue_length > 1 and upper.startswith("CE"):
        element = elem.carbon  # (probably) not Celenium...
    elif residue_length > 1 and upper.startswith("CD"):
        element = elem.carbon  # (probably) not Cadmium...
    elif residue_name in ["TRP", "ARG", "GLN", "HIS"] and upper.startswith("NE"):
        element = elem.nitrogen  # (probably) not Neon...
    elif residue_name in ["ASN"] and upper.startswith("ND"):
        element = elem.nitrogen  # (probably) not ND...
    elif residue_name == "CYS" and upper.startswith("SG"):
        element = elem.sulfur  # (probably) not SG...
    else:
        try:
            element = elem.get_by_symbol(atom_name[0])
        except KeyError:
            try:
                symbol = (
                    atom_name[0:2].strip().rstrip("AB0123456789").lstrip("0123456789")
                )
                element = elem.get_by_symbol(symbol)
            except KeyError:
                element = None

    return element


def _load_name_replacement_tables():
    PDBTrajectoryFile._loadNameReplacementTables()
    return (
        PDBTrajectoryFile._residueNameReplacements,
        PDBTrajectoryFile._atomNameReplacements,
    )


def _validate_pdb_file(filename):
    if filename.suffixes not in [[".pdb"], [".pdb", ".gz"]]:  # , [".cif"]]:
        raise ValueError(
            "PDB filename must have suffix `.pdb` or `.pdb.gz`"  # , or 'cif'. "
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
