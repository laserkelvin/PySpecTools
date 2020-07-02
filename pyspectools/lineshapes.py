import numpy as np
import numba

"""
    lineshapes.py

    Module for defining line shape functions. The two that
    are explicitly coded are Gaussian and Lorentizan functions,
    and derivatives of these are calculated analytically
    using SymPy.

    When available, use the lmfit built-in functions for
    lineshapes.
"""


@numba.jit(fastmath=True, nopython=True)
def gaussian(x: np.ndarray, A=1., x0=0., w=1.):
    """
    Vectorized implementation of a single Gaussian lineshape. For a given
    array of `x` values, we compute the corresponding amplitude of a
    Gaussian for a set of amplitude, center, and widths.
    
    Uses JIT compilation with Numba, and should be reasonably fast.

    Parameters
    ----------
    x : np.ndarray
        NumPy 1D array containing x values to evaluate over.
    A : float
        Scaling of the Gaussian; actually corresponds to the area.
    x0 : float
        Center of the Gaussian
    w : float
        Width of the Gaussian.

    Returns
    -------
    NumPy 1D array
        Array of amplitude values of the specified Gaussian
    """
    return A * np.exp(-(x - x0) ** 2.0 / (2.0 * w ** 2.0))


@numba.jit(fastmath=True, nopython=True)
def pair_gaussian(x: np.ndarray, A1: float, A2: float, x0: float, w: float, xsep: float):
    """
    Paired Gaussian lineshape. The function allows for the amplitudes to be
    floated, however the two Gaussians are locked in width and center
    frequency.

    Parameters
    ----------
    x : np.ndarray
        [description]
    A1, A2 : float
        Amplitude of the two Gaussians
    x0 : float
        Centroid of the two Gaussians
    w : float
        Width of the two Gaussians
    xsep : float
        Distance from the centroid and a Gaussian center

    Returns
    -------
    NumPy 1D array
        Array of amplitude values of the pair of Gaussians
    """
    return gaussian(x, A1, x0 - xsep, w) + gaussian(x, A2, x0 + xsep, w)


@numba.jit(fastmath=True, nopython=True)
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


@numba.jit(fastmath=True, nopython=True)
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


@numba.jit(fastmath=True, nopython=True)
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


@numba.jit(fastmath=True, nopython=True, parallel=True, nogil=True)
def fast_multi_gaussian(x: np.ndarray, A: np.ndarray, x0: np.ndarray, w: np.ndarray):
    """
    Fast, parallel implementation of a mixture of Gaussian lineshapes.
    Uses the `gaussian` function defined in this module, which itself is
    also JIT'd, and uses parallel loops to evaluate multiple Gaussians.
    
    The result is a NumPy 2D array of shape N x D, where N is the length
    of the frequency array, and D is the number of Gaussians.
    
    The inputs are expected to all be NumPy arrays, where A, x0, and w
    all equal in length and contain parameters for each respective Gaussian.
    
    With a 200,000 length frequency array and about 70 Gaussians, this code
    takes ~70 ms to compute; about four times faster than the unJIT'd version.

    Parameters
    ----------
    x : np.ndarray
        [description]
    A : np.ndarray
        [description]
    x0 : np.ndarray
        [description]
    w : np.ndarray
        [description]

    Returns
    -------
    [type]
        [description]
    """
    assert A.size == x0.size == w.size
    ngaussians = len(A)
    y = np.zeros((x.size, ngaussians))
    # parallelize the loop over number of Gaussians
    for i in numba.prange(ngaussians):
        y[:,i] = gaussian(x, A[i], x0[i], w[i])
    return y