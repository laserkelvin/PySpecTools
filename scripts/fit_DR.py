#!{python_path}

from pyspectools import doubleresonance as dr
import os

# Specify the filepath
filepath = input("Please specify the full file name.")
if os.path.isfile(filepath) is False:
    raise FileNotFoundError(filepath + " not found. Please try again.")

# Specify if a baseline correction is applied.
baseline_input = input("Fit baseline correction? Y/N")
if baseline_input.upper() == "Y":
    baseline = True
else:
    baseline= False

# Specify the frequency range to truncate. The default values will not
# truncate the spectrum at all.
valid = False
while valid is False:
    lowerfreq = input("Lower frequency cutoff? Default: 0.")

    if lowerfreq == "":
        lowerfreq = 0.
    try:
        lowerfreq = float(lowerfreq)
        valid = True
    except ValueError:
        print("Could not interpret frequency value. Please try again.")

valid = False
while valid is False:
    upperfreq = input("Lower frequency cutoff? Default: 1e9")

    if upperfreq == "":
        upperfreq = 1e9        # Some crazy number
    try:
        upperfreq = float(upperfreq)
        valid = True
    except ValueError:
        print("Could not interpret frequency value. Please try again.")

# Perform the analysis
dr.analyze_dr(filepath, baseline=baseline, freqrange=[lowerfreq, upperfreq])

