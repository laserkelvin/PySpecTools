

"""
    Wrapper functions for lmfit.

    Each module of PySpectools should use these functions
    when fitting is required, rather than rewrite for each
    purpose.
"""

import lmfit
import numpy as np
import pandas as pd

from pyspectools import lineshapes


class PySpecModel(lmfit.models.Model)
    def __init__(self, function, **kwargs):
        super.__init__(function, nan_policy="omit", **kwargs)
        self.params = self.make_params()


class FirstDerivLorentzian_Model(PySpecModel)


    """
    Child class of the PySpecModel, which in itself inherits from the `lmfit` `Models` class.
    Gives the first derivative Lorentzian line shape profile for fitting.
    """


def __init__(self, **kwargs):
    super.__init__(lineshapes.first_deriv_lorentzian, **kwargs)


class SecDerivLorentzian_Model(PySpecModel)


    """
    Child class of the PySpecModel, which in itself inherits from the `lmfit` `Models` class.
    Gives the second derivative Lorentzian line shape profile for fitting.
    """


def __init__(self, **kwargs):
    super.__init__(lineshapes.sec_deriv_lorentzian, **kwargs)


def rotor_energy(J, B, D=0.):
    """ Expression for a linear/prolate top with
        centrifugal distortion.

        parameters:
        ---------------
        J - integer quantum number
        B - rotational constant in MHz
        D - CD term in MHz

        returns:
        --------------
        state energy in MHz
    """
    return B * J * (J + 1) - D * J**2. * (J + 1)**2.


def calc_harmonic_transition(J, B, D=0.):
    """
        Calculate the transition frequency for
        a given upper state J, B, and D.

        parameters:
        --------------
        J - quantum number
        B - rotational constant in MHz
        D - centrifugal distortion constant in MHz

        returns:
        --------------
        transition frequency in MHz
    """
    lower = rotor_energy(J - 1, B, D)
    upper = rotor_energy(J, B, D)
    return upper - lower


def quant_check(value, threshold=0.001):
    """
        Function that will check if a value is close
        to an integer to the absolute value of the threshold.
        
        parameters:
        ---------------
        value - float for number to check
        threshold - float determining whether value is
                    close enough to being integer
                    
        returns:
        ---------------
        True if the value is close enough to being an integer,
        False otherwise.
    """
    nearest_half = np.round(value * 2) / 2
    return np.abs(nearest_half - value) <= threshold


def harmonic_fitter(progressions, J_thres=0.01):
    """
        Function that will sequentially fit every progression
        with a simple harmonic model defined by B and D. The
        "B" value here actually corresponds to B+C for a near-prolate,
        or 2B for a prolate top.
        
        There are a number of filters applied in order to minimize
        calculations that won't be meaningful - these parameters
        may have to be tuned for different test cases.
        
        Because the model is not actually quantized, J is
        represented as a float. To our advantage, this will
        actually separate real (molecular) progressions from
        fake news; at least half of the J values must be
        close to being an integer for us to consider fitting.
        
        parameters:
        ---------------
        progressions - iterable containing arrays of progressions
        J_thres - optional argument corresponding to how close a
                  value must be to an integer
                  
        returns:
        ---------------
        pandas dataframe containing the fit results; columns
        are B, D, fit RMS, and pairs of columns corresponding
        to the fitted frequency and approximate J value.
    """
    BJ_fit_model = lmfit.models.Model(calc_harmonic_transition)
    params = BJ_fit_model.make_params()
    data = list()
    progress = [0.25, 0.5, 0.75]
    progress = [int(value / len(progressions)) for value in progress]
    for index, progression in enumerate(progressions):
        if index in progress:
            print("{} progressions checked.".format(index))
        # Determine the approximate value of B based on
        # the differences between observed transitions
        approx_B = np.average(np.diff(progression))
        # Calculate the values of J that are assigned
        # based on B
        J = (progression / approx_B) / 2.
        # We want at least half of the lines to be
        # close to being integer
        if np.sum(quant_check(J, J_thres)) >= len(progression) / 1.5:
            # Let B vary a bit
            params["B"].set(
                approx_B,
                min=approx_B * 0.9,
                max=approx_B * 1.1
            )
            # Constrain D to be less than 5 MHz
            params["D"].set(
                0.001,
                min=0.,
                max=1.,
            )
            fit = BJ_fit_model.fit(
                data=progression,
                J=J,
                params=params,
                fit_kws={"maxfev": 50}
            )
            # Only include progressions that can be fit successfully
            if fit.success is True:
                # Calculate fit RMS
                rms = np.sqrt(np.average(np.square(fit.residual)))
                # Only add it to the list of the RMS is 
                # sufficiently low
                if rms < 50.:
                    return_dict = dict()
                    return_dict["RMS"] = rms
                    return_dict.update(fit.best_values)
                    # Make columns for frequency and J
                    for i, frequency in enumerate(progression):
                        return_dict[i] = frequency
                        return_dict["J{}".format(i)] = J[i]
                    data.append(return_dict)
    full_df = pd.DataFrame(
        data=data,
    )
    full_df.sort_values(["RMS", "B", "D"], ascending=False, inplace=True)
    return full_df
