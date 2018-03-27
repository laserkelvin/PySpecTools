
""" units.py

    Routines for performing unit conversions and quantities
    that are often used in spectroscopy.
"""

from scipy import constants

def kappa(A, B, C):
    # Ray's asymmetry parameter
    return (2*B - A - C) / (A - C)


def MHz2cm(frequency):
    # Convert frequency into wavenumbers
    return (frequency / 1000.) / (constants.c / 1e7)


def cm2MHz(wavenumber):
    # Convert wavenumbers to frequency in MHz
    return (wavenumber * (constants.c / 1e7)) * 1000.


