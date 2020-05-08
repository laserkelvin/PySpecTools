"""
test_assignment.py

PyTests to check the functionality of the `assignment` module
in `pyspectools.spectra`.

These tests use a pre-computed spectrum with known parameters,
and performs a series of assertions to make sure each step is
making sense.
"""

import pytest

from pyspectools.models import classes


def test_molecule_detective():
    constants = classes.SpecConstants(
        "5707.2341(231515)", "4032.2316(20)", "2620.21516(60)"
    )
    moldet = classes.MoleculeDetective()
    result = moldet(constants, composition=0, N=5000)
    result.analyze()
    