
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t uint8

"""
    routines.pyx
    
    Cython implementations of several routine functions where speed is
    possibly an issue.
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef isin_array(double[:] a, double[:] b, double tol=1e-2):
    """
    Function that will make piece-by-piece comparisons between two
    arrays, `a` and `b`. Every element in `a` will be checked for in `b`,
    and if there is an entry sufficiently close, it will return a True
    value for that element. 
    
    Conceptually, this function is similar to
    `np.isin()`, although it allows for two different sized arrays. The 
    resulting array will be length equal to `a`, while `b` can be
    larger or smaller than `a`.

    Parameters
    ----------
    a: 1D array
        Numpy 1D array to check
    b: 1D array
        Numpy 1D array to check for
    tol: float, optional
        Tolerance for proximity to check.

    Returns
    -------
    check_array: 1D array
        Numpy 1D array of length equal to `a`, with boolean values corresponding
        to whether element of `a` is in `b`.
    """
    cdef int i, j = 0
    cdef uint8 check = False
    cdef double a_val, b_val = 0.
    cdef int a_n = a.shape[0]
    cdef int b_n = b.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] check_array = np.zeros(
        a_n, dtype=np.uint8
    )
    # Outer loop on all of elements in a
    for i in range(a_n):
        # Get value of a
        a_val = a[i]
        # Inner loop on all of b
        for j in range(b_n):
            check = abs(a_val - b[j]) <= tol
            # See if values meet the tolerance
            if check == 1:
                check_array[i] = 1
            else:
                continue
    return check_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef hot_match_arrays(double[:] a, double[:] b, double tol=1e-2):
    """
    Function that will make piece-by-piece comparisons between two
    arrays, `a` and `b`. Every element in `a` will be checked for in `b`,
    and if there is an entry sufficiently close, it will make the 
    corresponding element 1.
    
    The result is a 2D array with the shape equal to `len(a)` x `len(b)`

    Parameters
    ----------
    a, b: 1D array
        Numpy 1D arrays to match
    tol: float, optional
        Tolerance for proximity to check.

    Returns
    -------
    check_array: 2D array
    """
    cdef int i, j = 0
    cdef uint8 check = False
    cdef double a_val, b_val = 0.
    cdef int a_n = a.shape[0]
    cdef int b_n = b.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=2] check_array = np.zeros(
        [a_n, b_n], dtype=np.uint8
    )
    # Outer loop on all of elements in a
    for i in range(a_n):
        # Get value of a
        a_val = a[i]
        # Inner loop on all of b
        for j in range(b_n):
            check = abs(a_val - b[j]) <= tol
            # See if values meet the tolerance
            if check == 1:
                check_array[i,j] = 1
            else:
                continue
    return check_array
