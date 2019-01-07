
import numpy as np

from pyspectools import units
from pyspectools.parsers import parse_cat
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
