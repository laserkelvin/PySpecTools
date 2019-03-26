
""" units.py

    Routines for performing unit conversions and quantities
    that are often used in spectroscopy.
"""

import numpy as np
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


def inertial_defect(A, B, C):
    """
    Calculate the inertial defect of a molecule with a set of A, B, and C rotational constants.

    Parameters
    ----------
    A, B, C - float
        Rotational constant along the respective principal axis in units of MHz.

    Returns
    -------
    inertial defect - float
    """
    return 505379. * (1 / C - 1 / B - 1 / A)


def rotcon2pmi(rotational_constant):
    """ Convert rotational constants in units of MHz to
        Inertia, in units of amu A^2.

        The conversion factor is adapted from:
        Oka & Morino, JMS (1962) 8, 9-21
        This factor comprises h / pi^2 c.

        :param rotational_constant: rotational constant in MHz
        :return:
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
    """
    Convert Hartrees to wavenumbers.
    :param hartree: float
    :return: corresponding value in 1/cm
    """
    return hartree * (harm / 100.)


def kjmol2wavenumber(kj):
    """
    Convert kJ/mol to wavenumbers
    :param kj: float
    :return: corresponding value in 1/cm
    """
    return kj * (jm / 100.) / (avo * 1000.)


def MHz2cm(frequency):
    """
    Convert MHz to wavenumbers
    :param frequency: float
    :return: corresponding value in 1/cm
    """
    return (frequency / 1000.) / (constants.c / 1e7)


def cm2MHz(wavenumber):
    """
    Convert wavenumbers to MHz
    :param wavenumber: float
    :return: corresponding value in MHz
    """
    return (wavenumber * (constants.c / 1e7)) * 1000.


def hartree2kjmol(hartree):
    """
    Convert Hartrees to kJ/mol.
    :param hartree: float
    :return: converted value in kJ/mol
    """
    return hartree * (eha * avo / 1000.)


def wavenumber2kjmol(wavenumber):
    """
    Convert wavenumbers to kJ/mol.
    :param wavenumber: float
    :return: converted value in kJ/mol
    """
    return wavenumber / (jm / 100.) / (avo * 1000.)


def T2wavenumber(T):
    """
    Convert temperature in Kelvin to wavenumbers.
    :param T: float
    :return: corresponding value in 1/cm
    """
    return T * kbcm


def wavenumber2T(wavenumber):
    """
    Convert wavenumbers to Kelvin
    :param wavenumber: float
    :return: corresponding value in K
    """
    return wavenumber / kbcm


""" 
    Astronomy units 

    Conversions and useful expressions
"""


def dop2freq(velocity, frequency):
    """
    Calculates the expected frequency in MHz based on a
    Doppler shift in km/s and a center frequency.
    :param velocity: float
    :param frequency: float
    :return: Doppler shifted frequency in MHz
    """
    # Frequency given in MHz, Doppler_shift given in km/s
    # Returns the expected Doppler shift in frequency (MHz)
    return ((velocity * 1000. * frequency) / constants.c)


def freq2vel(frequency, offset):
    """
    Calculates the Doppler shift in km/s based on a center
    frequency in MHz and n offset frequency in MHz (delta nu)
    :param frequency: float
    :param offset: float
    :return: Doppler shift in km/s
    """
    return ((constants.c * offset) / frequency) / 1000.


def gaussian_fwhm(sigma):
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


def gaussian_height(amplitude, sigma):
    """
        Calculate the height of a Gaussian distribution,
        based on the amplitude and sigma. This value
        corresponds to the peak height at the centroid.

        parameters:
        ----------------
        amplitude - float
        sigma - float

        returns:
        ----------------
        h - float
    """
    h = amplitude / (np.sqrt(2. * np.pi) * sigma)
    return h


def gaussian_integral(amplitude, sigma):
    """
    Calculate the integral of a Gaussian analytically using
    the amplitude and sigma.
    :param amplitude: amplitude of the Gaussian
    :param sigma: width of the Gaussian
    :return: integrated area of the Gaussian
    """
    integral = amplitude * np.sqrt(2. * np.pi**2. * sigma)
    return integral
