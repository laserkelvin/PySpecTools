
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
haev = constants.value("Hartree energy in eV")
hak = constants.value("hartree-kelvin relationship")
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
    delta - float
        The inertial defect in units of amu Ang**2
    """
    frac = np.reciprocal([C, B, A])
    cumdiff = frac[0]
    # Calculate the cumulative difference; i.e. 1/C - 1/B - 1/A
    for value in frac[1:]:
        cumdiff -= value
    return cumdiff * 505379.


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


def hartree2eV(hartree):
    """
    Convert Hartrees to eV.
    Parameters
    ----------
    hartree: float
        Electronic energy in Hartrees

    Returns
    -------
    eV: float
        Corresponding value in eV
    """
    return haev * hartree


def hartree2K(hartree):
    """
    Convert Hartrees to temperature in Kelvin.

    Parameters
    ----------
    hartree: float
        Electronic energy in Hartrees

    Returns
    -------
    kelvin: float
        Corresponding value in Kelvin
    """
    return hartree * hak


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


def thermal_corrections(frequencies, T, linear=True, hartree=True):
    """
    Calculates the thermal contributions from nuclear motion, in the same way as
    Gaussian does.
    Parameters
    ----------
    frequencies: list of floats or Numpy 1-D array
        Iterable of vibrational frequencies in wavenumbers. Can be Harmonic or Anharmonic fundamentals.
    T: float
        Value of the temperature to calculate thermal corrections at
    linear: bool, optional
        Specifies whether molecule is linear or not. Affects the rotational contribution.
    hartree: bool, optional
        If True, converts the contribution into Hartrees. Otherwise, in units of Kelvin

    Returns
    -------
    thermal: float
        Total thermal contribution at a given temperature
    """
    translation = units.kbcm * T
    rotation = units.kbcm * T
    if linear is False:
        rotation *= 3. / 2.
    # Convert frequencies into vibrational temperatures
    frequencies = np.asarray(frequencies)
    frequencies /= units.kbcm
    vibration = units.kbcm * np.sum(
        frequencies * (0.5 * (1. / (np.exp(frequencies / T) - 1.)))
    )
    thermal = translation + rotation + vibration
    if hartree is True:
        thermal *= 1. / units.hartree2wavenumber(1.)
    return thermal


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


def I2S(I, Q, frequency, E_lower, T=300.):
    """
    Function for converting intensity (in nm^2 MHz) to the more standard intrinsic linestrength, S_ij mu^2.

    Parameters
    ----------
    I - float
        The log of the transition intensity, typically given in catalog files
    Q - float
        Partition function at specified temperature T
    frequency - float
        Frequency of the transition in MHz
    E_lower - float
        ENergy of the lower state in wavenumbers
    T - float
        Temperature in Kelvin

    Returns
    -------
    siju - float
        Value of the intrinsic linestrength
    """
    E_upper = calc_E_upper(frequency, E_lower)
    # top part of the equation
    A = 10.**I * Q
    lower_factor = boltzmann_factor(E_lower, T)       # Boltzmann factors
    upper_factor = boltzmann_factor(E_upper, T)
    # Calculate the lower part of the equation
    # The prefactor included here is taken from Brian
    # Drouin's notes
    B = (4.16231e-5 * frequency * (lower_factor - upper_factor))
    return A / B


def S2I(S, Q, frequency, E_lower, T=300.):
    """
    Function for converting intensity (in nm^2 MHz) to the more standard intrinsic linestrength, S_ij mu^2.

    Parameters
    ----------
    S - float
        Intrinsic linestrength; Sij mu^2
    Q - float
        Partition function at specified temperature T
    frequency - float
        Frequency of the transition in MHz
    E_lower - float
        ENergy of the lower state in wavenumbers
    T - float
        Temperature in Kelvin

    Returns
    -------
    I - float
        log10 of the intensity at the specified temperature
    """
    E_upper = calc_E_upper(frequency, E_lower)
    lower_factor = boltzmann_factor(E_lower, T)       # Boltzmann factors
    upper_factor = boltzmann_factor(E_upper, T)
    # Calculate the lower part of the equation
    # The prefactor included here is taken from Brian
    # Drouin's notes
    B = (4.16231e-5 * frequency * (lower_factor - upper_factor))
    I = np.log((B * S) / Q)
    return I


def calc_E_upper(frequency, E_lower):
    """
    Calculate the upper state energy, for a given lower state energy and the frequency of the transition.

    Parameters
    ----------
    frequency - float
        Frequency of the transition in MHz
    E_lower - float
        Lower state energy in wavenumbers

    Returns
    -------
    E_upper - float
        Upper state energy in wavenumbers
    """
    transition_freq = MHz2cm(frequency)
    return transition_freq + E_lower


def calc_E_lower(frequency, E_upper):
    """
    Calculate the lower state energy, for a given lower state energy and the frequency of the transition.

    Parameters
    ----------
    frequency - float
        Frequency of the transition in MHz
    E_upper - float
        Upper state energy in wavenumbers

    Returns
    -------
    E_lower - float
        Lower state energy in wavenumbers
    """
    transition_freq = MHz2cm(frequency)
    return E_upper - transition_freq


def boltzmann_factor(E, T):
    """
    Calculate the Boltzmann weighting for a given state and temperature.

    Parameters
    ----------
    E - float
        State energy in wavenumbers
    T - float
        Temperature in Kelvin

    Returns
    -------
    boltzmann_factor - float
        Unitless Boltzmann factor for the state
    """
    return np.exp(-E / (kbcm * T))


def approx_Q_linear(B, T):
    """
    Approximate rotational partition function for a linear molecule.

    Parameters
    ----------
    B - float
        Rotational constant in MHz.
    T - float
        Temperature in Kelvin.

    Returns
    -------
    Q - float
        Rotational partition function at temperature T.
    """
    Q = 2.0837e4 * (T / B)
    return Q


def approx_Q_top(A, B, T, sigma=1, C=None):
    """
    Approximate expression for the (a)symmetric top partition function. The expression is adapted from Gordy and Cook,
    p.g. 57 equation 3.68. By default, the prolate top is used if the C constant is not specified, where B = C.
    Oblate case can also be specified if A = C.

    Parameters
    ----------
    A - float
        Rotational constant for the A principal axis, in MHz.
    B - float
        Rotational constant for the B principal axis, in MHz.
    T - float
        Temperature in Kelvin
    sigma - int
        Rotational level degeneracy; i.e. spin statistics
    C - float, optional
        Rotational constant for the C principal axis, in MHz. Defaults to None, which will reduce to the prolate
        top case.

    Returns
    -------
    Q - float
        Partition function for the molecule at temperature T
    """
    if C is None:
        # For a symmetric top, B = C
        C = B
    Q = (5.34e6 / sigma) * (T**3. / (A * B * C))**0.5
    return Q


def einsteinA(S, frequency):
    # Prefactor is given in the PGopher Intensity formulae
    # http://pgopher.chm.bris.ac.uk/Help/intensityformulae.htm
    # Units of the prefactor are s^-1 MHz^-3 D^-2
    # Units of Einstein A coefficient should be in s^-1
    prefactor = 1.163965505e-20
    return prefactor * frequency**3. * S
