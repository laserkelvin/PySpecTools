
import numpy as np
cimport numpy as np
cimport cython

from . import lineshapes


@cython.boundscheck(False)
def gaussian_filter(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y, float sigma):
    cdef int N = x.size
    cdef np.ndarray[np.float64_t, ndim=1] window = np.zeros(N)
    window = lineshapes.gaussian(x, 1., 0., sigma)
    window_fft = np.fft.rfft(window)
    y_fft = np.fft.rfft(y)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef matched_lorentzian(double[:] y, float w):
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
        xcorr: np.ndarray
            Numpy 1D array containing the x values.
    """
    cdef unsigned int i,j
    cdef unsigned int n = y.shape[0]
    cdef double integral = 0.
    cdef double[:] xcorr = np.zeros(n)
    cdef double[:] filt_y = np.zeros(n)
    cdef double[:] x = np.arange(n)
    for i in range(n):
        # Evaluate the lorentzian function
        filt_y = lineshapes.sec_deriv_lorentzian(x, <double>i, w, -1.)
        integral = 0.
        for j in range(n):
            integral += filt_y[j] * y[j]
        xcorr[i] = integral
    return xcorr
