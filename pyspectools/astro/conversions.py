
import numpy as np
from scipy import constants


def flux2N(W, Q, E, T, S, v):
    """
        Calculate the column density (1/cm^2) based on theoretical
        intensities for rotational transitions, and the observed
        flux per beam (Jy/beam)

        parameters:
        --------------
        W - integrated area of line profile; Jy
        Q - rotational partition function at given temperature
        E - upper state energy; Kelvin
        T - temperature; Kelvin
        S - formally Su^2; intrinsic linestrength
        v - transition frequency in MHz

        returns:
        --------------
        N - column density in 1/cm^2
    """
    numerator = 2.04 * W * Q * np.exp(E / T)
    denominator = S * (v / 1e3)**3.
    N = (numerator / denominator) * 10**20.
    return N


def N2flux(N, S, v, Q, E, T):
    """
        Calculate the expected integrated flux based on
        column density. This can be used to simulate a
        spectrum purely from theoretical terms.

        parameters:
        ---------------
        N - column density in cm^-2
        S - intrinsic line strength; Su^2
        v - transition frequency in MHz
        Q - rotational partition function
        E - state energy
        T - temperature

        returns:
        ---------------
        flux - the integrated flux in Jy
    """
    numerator = (N * S * (v / 1e3)**3.) / 1e20
    denominator = 2.04 * Q * np.exp(E / T)
    flux = numerator / denominator
    return flux
