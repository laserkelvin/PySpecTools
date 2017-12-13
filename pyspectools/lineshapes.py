
import numpy as np

"""
    lineshapes.py

    Module for defining line shape functions. The two that
    are explicitly coded are Gaussian and Lorentizan functions,
    and derivatives of these are calculated analytically
    using SymPy.
"""

def gaussian(x, A, c, w):
    # stock Gaussian distribution. The sign of A is such that the function
    # is flipped upside down.
    return A * np.exp(-(c - x)**2. / 2. * w**2.)


def lorentzian(x, x0, gamma, A):
    # Stock Lorentizan function
    return (A / np.pi * gamma) * (gamma**2. / ((x - x0)**2. + gamma**2.))


def first_deriv_lorentzian(x, x0, gamma, A):
    """
        Analytic first derivative function for a Lorentizan.
        This was solved using SymPy.
    """
    i = 0.63661977236758
    return ((i * A * gamma**3.0) * (x - x0)) / (gamma**2. + (x - x0)**2.)**2.


def sec_deriv_lorentzian(x, x0, gamma, A):
    """
        Analytic second derivative function for a Lorentizan.

        The function is split into two parts K and L.
    """
    i = 0.63661977236758
    j = 2.5464790894703
    K = -(i * A * gamma**3.) / (gamma**2. + (x - x0)**2.)**2.
    L = (j * A * gamma**3. * (x - x0)**2.) / (gamma**2. + (x - x0)**2.)**3.
    return K + L
