
import numpy as np


def lineprofile_analysis(fit, I, Q, T, E):
    """
        Low-level function to provide some analysis
        based on a fitted line profile and some theoretical
        parameters.

        parameters:
        ----------------
        fit - lmfit ModelResult object
        I - log10(theoretical intensity)
        Q - rotational partition function
        T - temperature in K
        E - lower state energy in K
    """
