"""
    assignment.py

    Contains dataclass routines for tracking assignments
    in broadband spectra.
"""

from dataclasses import dataclass, field
from lmfit.models import GaussianModel
from typing import List, Dict
import numpy as np


@dataclass
class Assignment:
    """
        DataClass for handling assignments.
        There should be sufficient information to store a
        line assignment and reproduce it later in a form
        that is both machine and human readable.

        parameters:
        ------------------
        name - str representing IUPAC/common name
        smiles - str representing SMILES code (specific!)
        frequency - float for observed frequency
        intensity - float for observed intensity
        peak_id - int for peak id from experiment
        composition - list-like with string corresponding to experimental
                      chemical composition. SMILES syntax.
        v_qnos - list with quantum numbers for vibrational modes. Index
                 corresponds to mode, and int value to number of quanta.
                 Length should be equal to 3N-6.
        experiment - int for experiment ID
        fit - dict-like containing the fitted parameters and model
    """
    name: str = ""
    smiles: str = ""
    frequency: float = 0.0
    intensity: float = 0.0
    peak_id: int = 0
    experiment: int = 0
    composition: List[str] = field(default_factory = list)
    v_qnos: List[int] = field(default_factory = list)
    r_qnos: List[int] = field(default_factory = list)
    fit: Dict = field(default_factory = dict)

    def __eq__(self, other):
        """ Dunder method for comparing molecules.
            This method is simply a shortcut to see if
            two molecules are the same based on their
            SMILES code.
        """
        return self.smiles == other

    def __str__(self):
        return f"{self.name}, {self.frequency}"

    def get_spectrum(self, x):
        """ Generate a synthetic peak by supplying
            the x axis for a particular spectrum. This method
            assumes that some fit parameters have been determined
            previously.

            parameters:
            ----------------
            x - 1D array with frequencies of experiment

            returns:
            ----------------
            y - 1D array of synthetic Gaussian spectrum
        """
        model = GaussianModel()
        params = model.make_params()
        params.update(self.fit)
        y = model.eval(params, x=x)
        return y

    @classmethod
    def from_dict(obj, data_dict):
        """ Method for generating an Assignment object
            from a dictionary. All this method does is
            unpack a dictionary into the __init__ method.

            parameters:
            ----------------
            data_dict - dict with DataClass fields

            returns:
            ----------------
            Assignment object
        """
        assignment_obj = obj(**data_dict)
        return assignment_obj

