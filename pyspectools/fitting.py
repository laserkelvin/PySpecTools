

"""
    Wrapper functions for lmfit.

    Each module of PySpectools should use these functions
    when fitting is required, rather than rewrite for each
    purpose.
"""


import numpy as np
import lmfit

from itertools import combinations

from pyspectools.lineshapes import first_deriv_lorentzian, sec_deriv_lorentzian


def construct_lineshape_mod(func_name="gaussian", n=1):
    """
        Serializes a model object from lmfit using the specified
        lineshape function.

        The arguments are string name of the function, and n for
        the number of lineshape functions to model.

        This function is generalized by hard-coding two cases:
        one where the functions are included in lmfit.models, and
        the other where they are not included, such as the second
        derivative line profiles.
    """
    # Convert to lowercase to make comparisons
    func_name = func_name.lower()
    # Available lineshape functions
    available_func = {
        "gaussian": lmfit.models.GaussianModel,
        "lorentzian": lmfit.models.LorentzianModel,
        "voigt": lmfit.models.VoigtModel,
        "first_lorentzian": first_deriv_lorentzian,
        "sec_lorentzian": sec_deriv_lorentzian,
    }
    # Check that the function is available
    if func_name not in available_func:
        raise KeyError("Lineshape function requested not available.")
    else:
        func = available_func[func]
    variable_names = func.func_code.co_varnames
    # Create a formattable index
    variable_names = [name + "_{index}" for name in variable_names]
    for index in range(n):
        # Case where non-derivatives are being used to fit
        # which can have a prefix to denote which function is which
        if "_" not in func_name:
            prefix = func_name + "_" + str(index)
            current_model = func(
                prefix=prefix
            )
        # For hard-coded functions
        else:
            variable_names = [name + "_{index}".format_map(index) for name in variable_names]
        current_params = current_model.make_params()
        # First case will initialize
        if index == 0:
            model = current_model
            parameters = current_params
        # All others will tack onto the model
        else:
            model+=current_model
            parameters+=current_params


def harmonic_molecule(J, B, D=0.):
    """ Expression for a linear/prolate top with
        centrifugal distortion.

        parameters:
        ---------------
        J - integer quantum number
        B - rotational constant in MHz
        D - CD term in MHz

        returns:
        --------------
        transition frequency in MHz
    """
    return B * J * (J + 1) - D * J**2. * (J + 1)**2.


def harmonic_fit(frequencies, maxJ=10, verbose=False):
    """
        Function for fitting a set of frequencies to a
        linear/prolate molecule model; i.e. B and D only.

        The primary function is for autofitting random peaks
        and seeing where there may be possible harmonic
        progressions in broadband spectra.

        Frequencies are sorted in ascending order, and then
        assigned a set of quantum numbers.

        It produces an approximate 2B value by taking half of 
        the average difference between frequencies.

        parameters:
        ------------------
        frequencies - iterable with floats corresponding to frequency
                      centers. Must be length greater than three.
        maxJ - optional int specifying maximum value of J

        returns:
        ------------------
        min_rms - minimum value for the rms from successful fits
        fit_values - dict with constants associated with the minimum rms fit
        fit_obj - ModelResult class with the best fit
    """
    frequencies = np.sort(frequencies)
    harm_model = lmfit.models.Model(harmonic_molecule)

    # Make guesses for constants based on frequencies
    approx_B = np.average(np.diff(frequencies))
    approx_D = np.std(np.diff(frequencies))

    # Set model parameters
    params = harm_model.make_params()
    params["B"].set(
        approx_B,
        min=0.,
        max=approx_B * 1.5
        )
    params["D"].set(
        approx_D,
        min=0.,
        max=approx_D * 10.
        )

    rms_bin = list()
    fit_values = list()
    fit_objs = list()

    # Generate every possible combination of quantum
    # numbers
    J_list = np.arange(1, maxJ)
    combo_obj = combinations(J_list, len(frequencies))

    if verbose:
        print("Estimated 2B: {:,.3f}".format(approx_B))
        print("Estimated D: {:,.3f}".format(approx_D))
        print("Number of combinations: {}".format(len(combo_obj)))
    # iterate over possible quantum number shifts
    for index, combo in enumerate(combo_obj):
        # Offset the frequency array by a shift
        result = harm_model.fit(
            frequencies,
            J=combo,
            params=params
            )
        # We only care about success stories
        if result.success is True:
            rms = np.sqrt(np.sum(np.square(result.residual)))
            rms_bin.append(rms)
            fit_values.append(result.best_values)
            fit_objs.append(result)
    min_rms = np.min(rms_bin)
    min_index = rms_bin.index(min_rms)
    return min_rms, min_index, rms_bin, fit_values, fit_objs
