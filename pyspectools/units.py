
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


def rotcon2pmi(rotational_constant):
    """ Convert rotational constants in units of MHz to
        Inertia, in units of amu A^2.

        The conversion factor is adapted from:
        Oka & Morino, JMS (1962) 8, 9-21
        This factor comprises h / pi^2 c.
    """
    return 1 / (rotational_constant / 134.901)


def inertial_defect(rotational_constants):
    """ Calculates the inertial defect, given three
        rotational constants in MHz. The ordering does not
        matter because the values are sorted.

        This value is I_c - I_a - I_b; a planar molecule
        is strictly zero.
    """
    # Ensure the ordering is A,B,C
    rotational_constants = np.sort(rotational_constants)[::-1]
    # Convert to principal moments of inertia
    pmi = rotcon2pmi(rotational_constants)
    return pmi[2] - pmi[0] - pmi[1]


def hartree2wavenumber(hartree):
    """ Convert Hartrees to wavenumbers """
    return hartree * (harm / 100.)


def kjmol2wavenumber(kj):
    # Convert kJ/mol to wavenumbers
    return kj * (jm / 100.) / (avo * 1000.)


def MHz2cm(frequency):
    # Convert frequency into wavenumbers
    return (frequency / 1000.) / (constants.c / 1e7)


def cm2MHz(wavenumber):
    # Convert wavenumbers to frequency in MHz
    return (wavenumber * (constants.c / 1e7)) * 1000.

""" 
    Astronomy units 

    Conversions and useful expressions
"""

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


def wavenumber2kjmol(wavenumber):
    return wavenumber / (jm / 100.) / (avo * 1000.)


def fwhm(sigma):
    """
        Calculate the full-width half maximum
        value assuming a Gaussian function.

        parameters:
        --------------
        sigma - float for width

        returns:
        --------------
        fwhm - float value for full-width at half-max
    """
    return 2. * np.sqrt(2. * np.log(2.)) * sigma

