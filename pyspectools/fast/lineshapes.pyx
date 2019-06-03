
import numpy as np
cimport numpy as np

from libc.math cimport sqrt, exp, pi


"""
    lineshapes.py

    Module for defining line shape functions. The two that
    are explicitly coded are Gaussian and Lorentizan functions,
    and derivatives of these are calculated analytically
    using SymPy.

    When available, use the lmfit built-in functions for
    lineshapes.
"""

cdef gaussian(np.ndarray[double, ndim=1] x, float A, float x0, float w):
    """
        Cython implementation of a Gaussian line shape. This function is
        designed to minimize Python interpreter interaction; everything
        except for the x values are C types, which should hopefully make
        it run fast.

        Parameters
        ----------
        x: np.ndarray
            1D array containing x values to evaluate
        A: float
            Amplitude of the Gaussian function
        x0: float
            Center value for the Gaussian function
        w: float
            Sigma of the Gaussian function

        Returns
        -------
        y: np.ndarray
            Numpy 1D array containing the x values.
    """
    cdef np.ndarray[double, ndim=1] y
    y = exp(-(x - x0)**2. / (2. * w**2.))
    y *= (A / sqrt(2. * pi * w**2.))
    return y


def pair_gaussian(np.ndarray [double, ndim=1] x, float A1, float A2,
        float x0, float w, float xsep):
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
    cdef np.ndarray[double, ndim=1] y1, y2
    y1 = gaussian(x, A1, x0 - xsep, w)
    y2 = gaussian(x, A2, x9 + sep, w)
    return y1 + y2


def lorentzian(np.ndarray[double, ndim=1] x, float x0, float gamma, float A):
    """
        Cython function for a Lorenztian distribution. Same as the corres-
        ponding Gaussian implementation, this function is written to minimize
        Python interpreter interaction if possible to optimize for speed.

        Parameters
        ----------
        x: np.ndarray
            Numpy 1D array of x values to evaluate on
        x0, gamma, A: float
            Parameters for the lorenztian function.
        
        Returns
        -------
        y: np.ndarray
            Numpy 1D array with y(x|x0, gamma, A)

    """
    cdef np.ndarray[double, ndim=1] y
    y = gamma**2. / ((x - x0)**2. + gamma**2.)
    y *= (A / (pi * gamma))
    return y


def first_deriv_lorentzian(np.ndarray [double, ndim=1] x, float x0, 
        float gamma, float A):
    """
        Cython implementation of the first derivative lineshape of a Lorenz-
        tian function.

        Parameters
        ----------
        x: np.ndarray
            Numpy 1D array containing x values to evaluate
        x0, gamma, A: float
            Parameters of the first-derivative Lorentzian function

        Returns
        -------
        y: np.ndarray
            Numpy 1D array containing y values for the first-derivative
    """
    cdef np.ndarray[double, ndim=1] y
    y = (-2. * A * gamma**3. * (x - x0))
    y /= (pi * (gamma**2. + (x - x0)**2.)**2.)
    return return y


def sec_deriv_lorentzian(np.ndarray [double, ndim=1] x, float x0, 
        float gamma, float A):
    """
        Cython implementation of the second derivative lineshape of a Lorenz-
        tian function.

        Parameters
        ----------
        x: np.ndarray
            Numpy 1D array containing x values to evaluate
        x0, gamma, A: float
            Parameters of the first-derivative Lorentzian function

        Returns
        -------
        y: np.ndarray
            Numpy 1D array containing y values for the second-derivative
    """
    cdef np.ndarray[double, ndim=1] y
    y = (-2. * A * gamma**3.) / (pi * (gamma**2. + (x - x0)**2.)**2.)
    y += (8. * A * gamma**3. * (x - x0)**2.) / (pi (gamma**2. + (x - x0_**2.)**3.))
    return y

