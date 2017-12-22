
import numpy as np
import lmfit

from lineshapes import first_deriv_lorentzian, sec_deriv_lorentzian


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
        
        current_params = current_model.make_params()
            # First case will initialize
            if index == 0:
                model = current_model
                parameters = current_params
            # All others will tack onto the model
            else:
                model+=current_model
                parameters+=current_params
    # For hard-coded functions
    else:
        variable_names = func.func_code.co_varnames
        # Create a formattable index
        variable_names = [name + "_{index}" for name in variable_names]
        for index in range(n):


def fit_lineshape(x, y, frequencies, lineshape_func):
    """
        Function for fitting individual lineshape to a spectrum using lmfit.
    """
