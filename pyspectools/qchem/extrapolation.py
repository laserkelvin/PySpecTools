
"""
extrapolation.py

Routines for performing SCF and correlation extrapolation.
"""

import numpy as np
from scipy.optimize import curve_fit as cf

"""
Extrapolation functions
"""

def linear(x, a, cbs):
    # For extrapolating CCSD(T) energy with linear form
    return cbs + (a / x**3)

def exponential(x, a, b, cbs):
    # For extrapolating the SCF energy with exponential formula
    return cbs + a * np.exp(-b * x)

def extrapolate_SCF(energies, basis):
    # Function for extrapolating the SCF contribution
    # to the CBS limit.
    # Takes a list of SCF energies in order
    # of T,Q,5-zeta quality, and returns the CBS energy.
    popt, pcov = cf(
        exponential,
        basis,
        energies,
        p0=[1., 1., np.min(energies)]
    )
    return popt[2], popt

def extrapolate_correlation(energies, basis):
    """
    Extrapolate the correlation energy to the CBS limit.
    Args: energies and basis are same length tuple-like, with the former
    corresponding to the correlation (post-SCF) energy and basis the cardinal
    values
    Returns: the extrapolated energy, and the results of the fit.
    """
    popt, pcov = cf(
        linear,
        basis,
        energies,
        p0=[1., np.min(energies)]
    )
    return popt[1], popt

