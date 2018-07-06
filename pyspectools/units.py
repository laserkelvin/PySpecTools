
""" units.py

    Routines for performing unit conversions and quantities
    that are often used in spectroscopy.
"""

from scipy import constants

""" Commonly used values

    kbcm - Boltzmann's constant in wavenumbers per Kelvin
    The others are pretty self-explanatory

    eha - Hartree energy in joules
"""
kbcm = constants.value("Boltzmann constant in inverse meters per kelvin") / 100.
avo = constants.Avogadro
eha = constants.value("Hartree energy")
harm = constants.value("hartree-inverse meter relationship")
jm = constants.value("joule-inverse meter relationship")


def kappa(A, B, C):
    # Ray's asymmetry parameter
    return (2*B - A - C) / (A - C)


def MHz2cm(frequency):
    # Convert frequency into wavenumbers
    return (frequency / 1000.) / (constants.c / 1e7)


def cm2MHz(wavenumber):
    # Convert wavenumbers to frequency in MHz
    return (wavenumber * (constants.c / 1e7)) * 1000.

""" Astronomy units """

def dop2freq(velocity, frequency):
    # Frequency given in MHz, Doppler_shift given in km/s
    # Returns the expected Doppler shift in frequency (MHz)
    return ((velocity * 1000. * frequency) / constants.c)


def freq2vel(frequency, offset):
    # Takes the expected Doppler contribution to frequency and the rest
    # frequency, and returns the Doppler shift in km/s
    return ((constants.c * offset) / frequency) / 1000.

def hartree2kjmol(hartree):
    # Convert Hartrees to kJ/mol
    return hartree * (eha * avo / 1000.)


def hartree2wavenumber(hartree):
    """ Convert Hartrees to wavenumbers """
    return hartree * (harm / 100.)


def kjmol2wavenumber(kj):
    # Convert kJ/mol to wavenumbers
    return kj * (jm / 100.) / (avo * 1000.)


def wavenumber2kjmol(wavenumber):
    return wavenumber / (jm / 100.) / (avo * 1000.)
