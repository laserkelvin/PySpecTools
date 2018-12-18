
import numpy as np
from scipy.integrate import quad

from pyspectools.lineshapes import gaussian
from pyspectools.units import gaussian_fwhm, gaussian_height, gaussian_integral
from pyspectools.astro import conversions
from pyspectools.astro import radiative


def lineprofile_analysis(fit, I, Q, T, E):
    """
        Low-level function to provide some analysis
        based on a fitted line profile and some theoretical
        parameters.

        Basically a collection of various parameters
        that are of interest to astronomy.

        parameters:
        ----------------
        fit - lmfit ModelResult object
        I - log10(theoretical intensity)
        Q - rotational partition function
        T - temperature in K
        E - lower state energy in K

        returns:
        -----------------
        data_dict - dict results of lineprofile analysis
    """
    # Calculate intrinsic line strength
    S = radiative.I2S(I, Q, fit.best_values["center"], E, T)
    # FWHM
    fwhm = gaussian_fwhm(fit.best_values["sigma"])
    # Analytic integration of a Gaussian
    integral = gaussian_integral(
        fit.best_values["amplitude"],
        fit.best_values["sigma"]
        )
    # Peak height of the Gaussian
    height = gaussian_height(
        fit.best_values["amplitude"],
        fit.best_values["sigma"]
        )
    # Column density
    N = conversions.flux2N(
        integral,
        Q,
        E,
        T,
        S,
        fit.best_values["center"]
        )
    data_dict = {
        "frequency": fit.best_values["center"],
        "peak height": height,
        "amplitude": fit.best_values["amplitude"],
        "width": fit.best_values["sigma"],
        "fwhm": fwhm,
        "integral": integral,
        "S $\mu^2$": S,
        "N cm$^{-2}$": N
        }
    return data_dict
    
