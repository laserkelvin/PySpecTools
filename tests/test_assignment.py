"""
test_assignment.py

PyTests to check the functionality of the `assignment` module
in `pyspectools.spectra`.

These tests use a pre-computed spectrum with known parameters,
and performs a series of assertions to make sure each step is
making sense.
"""

import os
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import yaml

from pyspectools.spectra import assignment


def test_spectrum_load():
    """
    This test makes sure that the spectrum parsing is done correctly.
    
    A Pandas DataFrame is read in "manually", and the same data is
    read in by `AssignmentSession.from_ascii`. The test then looks
    to make sure the number of elements (length of the dataframes)
    is equal, and then a numerical check to make sure all the rows
    are equal.
    """
    test_root = os.getcwd()
    # get dictionary with all of the simulated peak data
    with open("spectrum-parameters.yml", "r") as read_file:
        param_dict = yaml.load(read_file)
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # copy the spectrum into a temp directory, that way we don't
        # have to worry about cleaning up
        shutil.copy2("test-spectrum.csv", tmppath.joinpath("test-spectrum.csv"))
        spec_df = pd.read_csv("test-spectrum.csv")
        os.chdir(tmppath)
        # Create an AssignmentSession
        session = assignment.AssignmentSession.from_ascii(
            "test-spectrum.csv",
            0,
            skiprows=1,
            col_names=["Frequency", "Intensity"],
            delimiter=",",
        )
        # Check that the frequency and intensity columns are read correctly
        assert len(spec_df) == len(session.data)
        # this checks the whole dataframe is numerically equivalent
        assert spec_df.equals(session.data)
    os.chdir(test_root)


def test_spectrum_peaks():
    """
    This test checks whether or not the peak finding algorithm
    is functioning properly by comparing the number of actual
    peaks with the number found using `AssignmentSession.find_peaks`.
    
    There is a tolerance value used here, which basically gives some
    leeway to the peak finding which is seldom perfect. As long as
    the discrepancy is lower than `TOLERANCE`, then the test will pass.
    """
    test_root = os.getcwd()
    # The peak finding algorithm never finds every peak perfectly.
    # This TOLERANCE variable sets the minimum number to match
    TOLERANCE = 3
    # This is the minimum average deviation between the true
    # frequencies and the peak finding ones
    FREQ_TOLERANCE = 0.005
    # get dictionary with all of the simulated peak data
    with open("spectrum-parameters.yml", "r") as read_file:
        param_dict = yaml.load(read_file)
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # copy the spectrum into a temp directory, that way we don't
        # have to worry about cleaning up
        shutil.copy2("test-spectrum.csv", tmppath.joinpath("test-spectrum.csv"))
        os.chdir(tmppath)
        # Create an AssignmentSession
        session = assignment.AssignmentSession.from_ascii(
            "test-spectrum.csv",
            0,
            skiprows=1,
            col_names=["Frequency", "Intensity"],
            delimiter=",",
        )
        # Run the peak finding, with the intensity threshold set close
        # to the floor - I know where the noise should be :)
        peaks_df = session.find_peaks(2.0, als=False)
        # Make sure enough peaks are found; it doesn't have to be perfect
        assert abs(len(peaks_df) - param_dict["NPEAKS"]) <= TOLERANCE
    os.chdir(test_root)


def test_spectrum_linelist():
    """
    This test checks to make sure the `LineList` assignment process
    is working.
    
    The first test checks whether or not every line is
    assigned in the spectrum - there should be no more
    U-lines after this process is done, although there
    is a `TOLERANCE` specified that must be greater or
    equal to the number of U-lines remaining.
    
    The second test compares the frequency of every
    assignment against the "catalog" or actual frequency;
    the mean unsigned difference between must be equal to
    or below `FREQ_TOLERANCE`.
    """
    test_root = os.getcwd()
    # This TOLERANCE variable sets the minimum number of
    # assignments to match
    TOLERANCE = 3
    FREQ_TOLERANCE = 0.002
    # get dictionary with all of the simulated peak data
    with open("spectrum-parameters.yml", "r") as read_file:
        param_dict = yaml.load(read_file)
        sub_dict = {key: param_dict[key] for key in ["AMPS", "CENTERS", "ASSIGNMENTS"]}
        assignment_df = pd.DataFrame(sub_dict)
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # copy the spectrum into a temp directory, that way we don't
        # have to worry about cleaning up
        shutil.copy2("test-spectrum.csv", tmppath.joinpath("test-spectrum.csv"))
        os.chdir(tmppath)
        # Create an AssignmentSession
        session = assignment.AssignmentSession.from_ascii(
            "test-spectrum.csv",
            0,
            skiprows=1,
            col_names=["Frequency", "Intensity"],
            delimiter=",",
        )
        # Run the peak finding, with the intensity threshold set close
        # to the floor - I know where the noise should be :)
        peaks_df = session.find_peaks(2.0, als=False)
        # This is the number of peaks found; assignment comparisons
        # should be made with this value
        NFOUND = len(peaks_df)
        # Generate LineList objects for each peak group
        for group in assignment_df["ASSIGNMENTS"].unique():
            subgroup = assignment_df.loc[assignment_df["ASSIGNMENTS"] == group]
            linelist = assignment.LineList.from_list(
                f"Group{group}", subgroup["CENTERS"].tolist()
            )
            # PRIOR = len(session.line_list["Peaks"].get_ulines())
            session.process_linelist(linelist=linelist)
        assignments = session.line_lists["Peaks"].get_assignments()
        POST = len(assignments)
        # Compare the number of assignments made vs. the number of peaks
        # Every peak should be assigned
        assert abs(POST - NFOUND) <= TOLERANCE
        # Check that the frequency deviation between catalog and matched
        # is below a threshold
        peak_freqs = np.array([getattr(line, "frequency") for line in assignments])
        actual_freqs = np.array([getattr(line, "catalog_frequency") for line in assignments])
        AVG_DEVIATION = np.mean(np.abs(peak_freqs - actual_freqs))
        assert AVG_DEVIATION <= FREQ_TOLERANCE
    os.chdir(test_root)

