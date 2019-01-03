
import numpy as np

"""
    lineshapes.py

    Module for defining line shape functions. The two that
    are explicitly coded are Gaussian and Lorentizan functions,
    and derivatives of these are calculated analytically
    using SymPy.

    When available, use the lmfit built-in functions for
    lineshapes.
"""

def gaussian(x, A, x0, w):
    # stock Gaussian distribution. The sign of A is such that the function
    # is flipped upside down.
    return A * np.exp(-(x - x0)**2. / (2. * w**2.))


def pair_gaussian(x, A1, A2, x0, w, xsep):
    """
    Paired Gaussian lineshape. The function allows for the amplitudes to be
    floated, however the two Gaussians are locked in width and center
    frequency.
    :param x: np.array to evaluate the pair gaussians on
    :param A1, A2: float amplitude of the gaussians
    :param x0: float center-of-mass value for the two gaussian centers
    :param w: float width of the gaussians
    :param xsep: float seperation expected for the gaussian centers
    :return: np.array y values of the gaussian
    """
    return gaussian(x, A1, x0 - xsep, w) + gaussian(x, A2, x0 + xsep, w)


def lorentzian(x, x0, gamma, A):
    # Stock Lorentizan function
    return (A / np.pi * gamma) * (gamma**2. / ((x - x0)**2. + gamma**2.))


def first_deriv_lorentzian(x, x0, gamma, A):
    """
        Analytic first derivative function for a Lorentizan.
        This was solved using SymPy.
    """
    return (-2. * A * gamma**3. * (x - x0)) / (np.pi * (gamma**2. + (x - x0)**2.)**2.)


def sec_deriv_lorentzian(x, x0, gamma, A):
    """
        Analytic second derivative function for a Lorentizan.
        Similar to the first derivative, this was derived using SymPy.
        The function is split into two parts K and L.
    """
    K = (-2. * A * gamma**3.) / (np.pi * (gamma**2. + (x - x0)**2.)**2.)
    L = (8. * A * gamma**3. * (x - x0)**2.) / (np.pi (gamma**2. + (x - x0_**2.)**3.))
    return K + L
