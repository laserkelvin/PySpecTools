"""
    Wrapper functions for lmfit.

    Each module of PySpectools should use these functions
    when fitting is required, rather than rewrite for each
    purpose.
"""

import lmfit
import numpy as np
import pandas as pd
import peakutils
from tqdm.autonotebook import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pyspectools import lineshapes


class PySpecModel(lmfit.models.Model):
    def __init__(self, function, **kwargs):
        super(PySpecModel, self).__init__(function, **kwargs)
        self.params = self.make_params()


class FirstDerivLorentzian_Model(PySpecModel):
    """
    Child class of the PySpecModel, which in itself inherits from the `lmfit` `Models` class.
    Gives the first derivative Lorentzian line shape profile for fitting.
    """

    def __init__(self, **kwargs):
        super(FirstDerivLorentzian_Model, self).__init__(
            lineshapes.first_deriv_lorentzian, **kwargs
        )


class SecDerivLorentzian_Model(PySpecModel):
    """
    Child class of the PySpecModel, which in itself inherits from the `lmfit` `Models` class.
    Gives the second derivative Lorentzian line shape profile for fitting.
    """

    def __init__(self, **kwargs):
        super(SecDerivLorentzian_Model, self).__init__(
            lineshapes.sec_deriv_lorentzian, **kwargs
        )


class BJModel(PySpecModel):
    """
    Model for fitting prolate/linear molecules.
    """

    def __init__(self, **kwargs):
        super(BJModel, self).__init__(calc_harmonic_transition, **kwargs)


class PairGaussianModel(PySpecModel):
    def __init__(self, **kwargs):
        super(PairGaussianModel, self).__init__(
            lineshapes.pair_gaussian, independent_vars=["x"], **kwargs
        )

    def fit_pair(self, x, y, verbose=False):
        # Automatically find where the Doppler splitting is
        indexes = peakutils.indexes(y, thres=0.4, min_dist=10)
        best_two = np.argsort(y[indexes])
        guess_center = np.average(x[best_two])
        guess_sep = np.std(x[best_two])
        # This calculates the amplitude of a Gaussian based on
        # the peak height
        prefactor = np.sqrt(2.0 * np.pi) * 0.01
        guess_amp = np.average(y[best_two]) * prefactor
        # Set the parameter guesses
        self.params["A1"].set(guess_amp)
        self.params["A2"].set(guess_amp)
        self.params["w"].set(0.005, min=0.0001, max=0.03)
        if np.isfinite(guess_sep) is True:
            self.params["xsep"].set(guess_sep, min=guess_sep * 0.8, max=guess_sep * 1.2)
        else:
            self.params["xsep"].set(0.05)
        self.params["x0"].set(
            guess_center, min=guess_center - 0.6, max=guess_center + 0.6
        )
        if verbose is True:
            print("Peaks found: {}".format(x[best_two]))
            print("Initial parameters:")
            for key, value in self.params.items():
                print(key, value)
        results = self.fit(data=y, x=x, params=self.params)
        return results


def rotor_energy(J, B, D=0.0):
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
    return B * J * (J + 1) - D * J ** 2.0 * (J + 1) ** 2.0


def calc_harmonic_transition(J, B, D=0.0):
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
    fit_objs = list()
    failed = list()
    for index, progression in tqdm(enumerate(progressions)):
        # Determine the approximate value of B based on
        # the differences between observed transitions
        approx_B = np.average(np.diff(progression))
        # Calculate the values of J that are assigned based on B
        J = (progression / approx_B) / 2.0
        # We want at least half of the lines to be close to being integer
        if len(progression) >= 2:
            if np.sum(quant_check(J, J_thres)) >= len(progression) / 1.5:
                # Let B vary a bit
                params["B"].set(approx_B, min=approx_B * 0.9, max=approx_B * 1.1)
                # Constrain D to be less than 5 MHz
                params["D"].set(0.001, min=0.0, max=1.0)
                fit = BJ_fit_model.fit(
                    data=progression, J=J, params=params, fit_kws={"maxfev": 100}
                )
                # Only include progressions that can be fit successfully
                if fit.success is True:
                    # Calculate fit RMS
                    rms = np.sqrt(np.average(np.square(fit.residual)))
                    # Only add it to the list of the RMS is
                    # sufficiently low
                    return_dict = dict()
                    return_dict["RMS"] = rms
                    return_dict.update(fit.best_values)
                    # Make columns for frequency and J
                    for i, frequency in enumerate(progression):
                        return_dict[i] = frequency
                        return_dict["J{}".format(i)] = J[i]
                    data.append(return_dict)
                    fit_objs.append(fit)
                else:
                    failed.append([index, fit.fit_report()])
            else:
                failed.append(index)
        else:
            return_dict = dict()
            return_dict["RMS"] = 0.0
            return_dict["B"] = approx_B
            # reformat the frequencies and approximate J values
            for i, frequency in enumerate(progression):
                return_dict[i] = frequency
                return_dict["J{}".format(i)] = J[i]
            data.append(return_dict)
    full_df = pd.DataFrame(data=data)
    full_df.sort_values(["RMS", "B", "D"], ascending=False, inplace=True)
    return full_df, fit_objs


def baseline_als(y, lam=1e4, p=0.01, niter=10):
    """
        Function for performing an iterative baseline
        fitting using the asymmetric least squares algorithm.

        This refers to the paper:
        "Baseline Correction with Asymmetric Least Squares Smoothing"
        by Eilers and Boelens (2005).

        The code is taken from a Stack Overflow question:
        https://stackoverflow.com/a/29185844

        According to the paper, the tested values are:
        0.001 <= p <= 0.1 for positive peaks
        1e2 <= lam <= 1e9

        Parameters:
        --------------
        y : numpy 1D array
            Data used to fit the baseline
        lam : float
            Tuning factor for penalty function that offsets the difference cost function
        p : float
            Weighting factor for the cost function

        Returns:
        --------------
        z : numpy 1D array
            Array containing the baseline values
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    # Initialize a set of weights
    w = np.ones(L)
    # Iterate for a set number of times to fit baseline
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z
