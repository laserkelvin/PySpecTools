
import numpy as np
from lmfit.models import LinearModel

from pyspectools import units
from pyspectools.parsers import parse_cat
from pyspectools.units import gaussian_fwhm, gaussian_height, gaussian_integral
from pyspectools.astro import conversions
from pyspectools.astro import radiative


def lineprofile_analysis(fit, I, Q, T, E_lower):
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
    S = radiative.I2S(I, Q, fit.best_values["center"], E_lower, T)
    E_upper = radiative.calc_E_upper(
        fit.best_values["center"],
        E_lower
    )
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
    L = calculate_L(integral, fit.best_values["center"], S)
    data_dict = {
        "frequency": fit.best_values["center"],
        "peak height": height,
        "amplitude": fit.best_values["amplitude"],
        "width": fit.best_values["sigma"],
        "fwhm": fwhm,
        "integral": integral,
        "E upper": E_upper,
        "S $\mu^2$": S,
        "N cm$^{-2}$": N,
        "L": L
        }
    return data_dict


def simulate_catalog(catalogpath, N, Q, T, doppler=10.):
    """
     Function for simulating what the expected flux would be for
     a given molecule and its catalog file from SPCAT. Returns a
     spectrum of Gaussian line shapes, predicted by either a supplied
     Doppler width or session-wide value (self.doppler), and
     the parameters required to calculate the flux in Jy.

     :param catalogpath: path to the SPCAT catalog file
     :param N: column density in cm^-2
     :param Q: rotational partition function
     :param T: temperature in Kelvin
     :param doppler: Doppler width in km/s
     :return: simulated_df; pandas dataframe with frequency/intensity
     """
    catalog_df = parse_cat(catalogpath)
    # Calculate the line strength
    catalog_df["Su^2"] = radiative.I2S(
        catalog_df["Intensity"].values,
        Q,
        catalog_df["Frequency"].values,
        catalog_df["Lower state energy"].values,
        T
    )
    # Calculate the expected integrated flux in Jy
    catalog_df["Integrated Flux (Jy)"] = conversions.N2flux(
        N,
        catalog_df["Su^2"].values,
        catalog_df["Frequency"].values,
        Q,
        catalog_df["Lower state energy"].values,
        T
    )
    catalog_df["Doppler Shifts"] = units.dop2freq(velocity=doppler, frequency=catalog_df["Frequency"].values)
    catalog_df["Doppler Frequency"] = catalog_df["Frequency"] + catalog_df["Doppler Shifts"]
    amplitudes = catalog_df["Integrated Flux (Jy)"] / np.sqrt(2. * np.pi**2. * catalog_df["Doppler Shifts"])
    catalog_df["Flux (Jy)"] = units.gaussian_height(
        amplitudes,
        catalog_df["Doppler Shifts"]
    )
    return catalog_df


def calculate_L(W, frequency, S):
    """
    Calculate L, which is subsequently used to perform rotational temperature analysis.

    Parameters
    ----------
    W - float
        Integrated flux of a line
    frequency - float
        Frequency of a transition in MHz
    S - float
        Intrinsic linestrength of the transition

    Returns
    -------
    L - float
    """
    L = (2.04e20 * W) / (S * frequency**3)
    return L


def rotational_temperature_analysis(L, E_upper):
    """
    Function that will perform a rotational temperature analysis. This will perform a least-squares fit of log(L),
    which is related to the theoretical line strength and integrated flux, and the upper state energy for the same
    transition.

    Parameters
    ----------
    L - 1D array
        Value L related to the line and theoretical line strength
    E_upper - 1D array
        The upper state energy in wavenumbers.

    Returns
    -------
    ModelResult - object
        Result of the least-squares fit
    """
    # Convert the upper state energy
    E_upper *= units.kbcm
    logL = np.log(L)
    model = LinearModel()
    params = model.make_params()
    result = model.fit(
        x=E_upper,
        y=logL,
        params=params
    )
    return result
