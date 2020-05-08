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
    return A * np.exp(-(x - x0) ** 2.0 / (2.0 * w ** 2.0))


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


def lorentzian(x, x0, gamma, I):
    """
    Function to evaluate a Lorentzian lineshape function.
    
    Parameters
    ----------
    x : Numpy 1D array
        Array of floats corresponding to the x values to evaluate on
    x0 : float
        Center for the distribution
    gamma : float
        Width of the distribution
    I : float
        Height of the distribution
    
    Returns
    -------
    Numpy 1D array
        Values of the Lorentzian distribution
    """
    return I * (gamma ** 2.0 / ((x - x0) ** 2.0 + gamma ** 2.0))


def first_deriv_lorentzian(x, x0, gamma, I):
    """
    Function to evaluate the first derivative of a Lorentzian lineshape function.
    
    Parameters
    ----------
    x : Numpy 1D array
        Array of floats corresponding to the x values to evaluate on
    x0 : float
        Center for the distribution
    gamma : float
        Width of the distribution
    I : float
        Height of the distribution
    
    Returns
    -------
    Numpy 1D array
        Values of the Lorentzian distribution
    """
    return (
        -2.0
        * I
        * gamma ** 2.0
        * (x - x0) ** 1.0
        / (gamma ** 2.0 + (x - x0) ** 2.0) ** 2
    )


def sec_deriv_lorentzian(x, x0, gamma, I):
    """
    Function to evaluate the second derivative of a Lorentzian lineshape function.
    This was evaluated analytically with SymPy by differentiation of the
    Lorentzian expression used for the `lorentzian` function in this module.
    
    Parameters
    ----------
    x : Numpy 1D array
        Array of floats corresponding to the x values to evaluate on
    x0 : float
        Center for the distribution
    gamma : float
        Width of the distribution
    I : float
        Height of the distribution
    
    Returns
    -------
    Numpy 1D array
        Values of the Lorentzian distribution
    """
    return (
        -I
        * gamma ** 2.0
        * (2.0 - 8.0 * (x - x0) ** 2.0 / (gamma ** 2.0 + (x - x0) ** 2.0))
        / (gamma ** 2.0 + (x - x0) ** 2.0) ** 2
    )
