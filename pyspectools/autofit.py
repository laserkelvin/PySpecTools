"""
    Routines for performing automatic fitting using
    SPFIT/SPCAT.
"""

import numpy as np
from pyspectools import routines
from collections import OrderedDict


class Autofit:
    """
        Class for an Autofit object which handles
        all of the formatting and executable calls.
    """

    def __init__(self, input_dict):
        # Initialize the default settings for the Autofit
        # class.
        self.inp = {
            "nqn": 4,
            "min_N": 0,
            "max_N": 2,
            "min_Ka": 0,
            "max_Ka": 0,
            "min_Kc": 0,
            "max_Kc": 2,
            "max_hfs": 10,
            "nuclei": 0,
            "electron": 0,
            "frequencies": list(),
            "mode": "brute",
            "fixed": dict(),
        }
        # Overwrite dictionary with specified values
        self.inp.update(input_dict)

    @classmethod
    def from_yaml(cls, filepath):
        """ Class method to initialize an Autofit obj
            by reading in a YAML syntax file.

            parameters:
            ------------------
            filepath - str specifying path to yaml file

            returns:
            ------------------
            autofit obj
        """
        yaml_dict = routines.read_yaml(filepath)
        return cls(yaml_dict)

    def generate_qnos(self):
        """
            Function for generating random quantum numbers based on
            what is allowed.

            These rules are hardcoded because certain aspects are
            already pre-ordained: e.g.
        """
        N_u = np.random.randint(self.inp["min_N"], self.inp["max_N"] + 1)
        Ka_u = np.random.randint(self.inp["min_Ka"], self.inp["max_Ka"] + 1)
        Kc_u = np.random.randint(self.inp["min_Kc"], self.inp["max_Kc"] + 1)
        # Generate lower state quantum numbers
        N_l = delta_no(N_u, True)
        Ka_l = delta_no(Ka_u, True)
        Kc_l = delta_no(Kc_u, True)
        # Store quantum numbers in an ordered dictionary
        qnos = OrderedDict(
            {
                "N_u": N_u,
                "Ka_u": Ka_u,
                "Kc_u": Kc_u,
                "N_l": N_l,
                "Ka_l": Ka_l,
                "Kc_l": Kc_l,
            }
        )
        # Generate the funky stuff
        if self.inp["nuclei"] > 0:
            F_u = np.random.randint(0, self.inp["max_hfs"] + 1)
            # Generate lower level hf qno; delta F = -1, 0, +1
            F_l = delta_no(F_u)

        return None

    def delta_no(self, value, N=False):
        """ Simple function to generate a lower state quantum
            number based on the selection rule provided.

            The value must also be greater or equal to zero.

            For example, delta N = -1/0
            delta K = -1, 0, 1
            delta F = -1, 0, 1

            parameters:
            -----------------
            value - integer value of upper state quantum number
            N - bool arg specifying whether the quantum number is N

            returns:
            -----------------
            lower_qno - value for lower quantum number
        """
        if N == True:
            upper = 2
        else:
            upper = 1
        lower_qno = -1
        # Make sure the value is greater than zero
        while lower_qno >= 0:
            lower_qno = value + random.randint(-1, upper)
        return lower_qno

    class Line:
        """
            Line object that holds the frequency and
            assignments. Has access to methods for
            formatting.
        """

        def __init__(self, frequency, unc=0.002):
            self.freq = float(frequency)
            self.unc = unc

        def __str__(self):
            # Formats the frequency and uncertainty to the
            # usual value(uncertainty) format.
            form_str = routines.format_uncertainty(self.freq, self.unc)
            return form_str

        def set_assignments(self, assignment):
            """
                Function for setting up the quantum number
                assignments.

                The assignments come in an OrderedDict such that
                when we pipe the data back out we want the upper
                levels to come out first.

                parameters:
                ----------------
                assignment- key/value pairs correspond to the quantum
                            number and value. Must be an OrderedDict
            """
            assert type(assignment) == OrderedDict
            self.assignment = assignment

        def format_lin(self):
            """ Function for formatting the line into a .lin
                file format.
                12I3,freeform
                QN,Freq,Err,Wt
            """
            format_str = ""
            for key, value in self.assignments.items():
                format_str += "{: >3d}".format(value)
            format_str += " "
            format_str += "{:.4}".format(self.freq)
            format_str += " "
            format_str += "{:.4}".format(self.unc)
            format_str += " "
            format_str += "1E-5"
            return format_str
