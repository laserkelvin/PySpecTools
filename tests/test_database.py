
from tempfile import NamedTemporaryFile

from pytest import approx

from pyspectools.database import MoleculeDatabase


def test_db_creation():
    temp_file = NamedTemporaryFile()
    entry = {"garbage": 1}
    with MoleculeDatabase(temp_file.name) as mol_db:
        mol_db.insert(entry)


def test_db_io():
    temp_file = NamedTemporaryFile()
    with MoleculeDatabase(temp_file.name) as mol_db:
        mol_db.add_molecule_yaml("test_mol.yml")
        # now retrieve the entry
        entry = mol_db.get_smiles("C1=COC=C1")
        mol, spcat = mol_db.to_molecule(entry)
    assert mol.num_nuclei == 0
    assert mol.A.value == 9447.12422
    assert spcat.mu[0] == 0.661
    # try and run SPCAT now
    spcat.run(mol)
