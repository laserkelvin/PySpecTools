
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

