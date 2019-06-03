
import numpy as np
cimport numpy as np
cimport cython

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef gaussian(double[:] x, double A, double x0, double w):
    """
        Cython implementation of a Gaussian line shape. This function is
        designed to minimize Python interpreter interaction; everything
        except for the x values are C types, which should hopefully make
        it run fast.
        
        This function holds the advantage of static typing over the NumPy
        implementation in the `pyspectools.lineshapes` module. From rudimentary
        timing tests, the best improvement is seen for arrays <1000 elements,
        from which the NumPy version is faster.

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
    cdef unsigned int index
    cdef unsigned int n = x.shape[0]
    cdef double[:] y = np.zeros(n)
    for index in range(n):
        y[index] = exp(-(x[index] - x0)**2. / (2. * w**2.))
        y[index] *= (A / sqrt(2. * pi * w**2.))
    #y = exp(-(x - x0)**2. / (2. * w**2.))
    #y *= (A / sqrt(2. * pi * w**2.))
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef multi_gaussian(double[:] x, double[:] A, double[:] x0, double[:] w):
    """
    Sum of multiple Gaussian distributions written in Cython. This version of the code is written so that
    there should be no overhead associated with NumPy within the loop. The array access is completely within
    a single function, rather than repeatedly calling the `gaussian` function. In theory this should be a
    faster implementation.
    
    Parameters
    ----------
    x: array_like
        1D Numpy array of floats corresponding to the x values to evaluate
    A, x0, w: array_like
        1D Numpy array of floats corresponding to the amplitudes, centers, and widths of the Gaussians

    Returns
    -------
    y: array_like
        Numpy 1D array of Y(X|A,X0,W)
    """
    cdef unsigned int i, j
    cdef unsigned int n = x.size
    cdef unsigned int m = x0.size
    cdef double[:] y = np.zeros(n)
    # Make sure that the size of the parameter arrays are equal
    assert A.size == x0.size == w.size

    for i in range(n):
        for j in range(m):
            y[i] += (A[j] / sqrt(2. * pi * w[j]**2.))
            y[i] *= exp(-(x[i] - x0[j])**2. / (2. * w[j]**2.))
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef pair_gaussian(double[:] x, float A1, float A2, float x0, float w, float xsep):
    """
    Pair Gaussian function. 
    Parameters
    ----------
    x: array_like
        Numpy 1D array of X values to evaluate the pair of Gaussians on
    A1, A2: float
        Amplitudes of the two Gaussians
    x0: float
        Center-of-mass for the two Gaussians
    w: float
        Width of the two gaussians
    xsep: float
        Separation in units of x that pushes the two Gaussians from the center-of-mass

    Returns
    -------
    y: array_like
        Numpy 1D array of Y values
    """
    cdef unsigned int index
    cdef unsigned int n = x.shape[0]
    cdef double[:] y = np.zeros(n)
    for index in range(n):
        y[index] = exp(-(x[index] - x0 + xsep)**2. / (2. * w**2.)) * (A1 / sqrt(2. * pi * w**2.))
        y[index] += exp(-(x[index] - x0 - xsep)**2. / (2. * w**2.)) * (A2 / sqrt(2. * pi * w**2.))
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef lorentzian(double[:] x, float x0, float gamma, float A):
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
    cdef unsigned int n = x.size
    cdef unsigned int i
    cdef double[:] y = np.zeros(n)
    for i in range(n):
        y[i] = gamma**2. / ((x[i] - x0)**2. + gamma**2.)
        y[i] *= (A / (pi * gamma))
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef first_deriv_lorentzian(double[:] x, float x0, float gamma, float A):
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
    cdef unsigned int n = x.size
    cdef unsigned int i
    cdef double[:] y = np.zeros(n)
    for i in range(n):
        y[i] = (-2. * A * gamma**3. * (x[i] - x0))
        y[i] /= (pi * (gamma**2. + (x[i] - x0)**2.)**2.)
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef sec_deriv_lorentzian(double[:] x, float x0, float gamma, float A):
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
    cdef unsigned int n = x.size
    cdef unsigned int i
    cdef double[:] y = np.zeros(n)
    for i in range(n):
        y[i] = (-2. * A * gamma**3.) / (pi * (gamma**2. + (x[i] - x0)**2.)**2.)
        y[i] += (8. * A * gamma**3. * (x[i] - x0)**2.) / (pi * (gamma**2. + (x[i] - x0**2.)**3.))
    return y
