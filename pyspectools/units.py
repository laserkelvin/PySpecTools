
""" units.py

    Routines for performing unit conversions and quantities
    that are often used in spectroscopy.
"""

from scipy import constants

""" Commonly used values

    kbcm - Boltzmann's constant in wavenumbers per Kelvin
"""
kbcm = constants.value("Boltzmann constant in inverse meters per kelvin") / 100.


def kappa(A, B, C):
    # Ray's asymmetry parameter
    return (2*B - A - C) / (A - C)


def MHz2cm(frequency):
    # Convert frequency into wavenumbers
    return (frequency / 1000.) / (constants.c / 1e7)


def cm2MHz(wavenumber):
    # Convert wavenumbers to frequency in MHz
    return (wavenumber * (constants.c / 1e7)) * 1000.


def hartree2kjmol(hartree):
    # Convert Hartrees to kJ/mol
    return hartree * 2625.499638


def kjmol2wavenumber(kj):
    # Convert kJ/mol to wavenumbers
    return kj * 83.593


def wavenumber2kjmol(wavenumber):
    return wavenumber / 83.59

