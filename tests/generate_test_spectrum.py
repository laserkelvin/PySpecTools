"""
generate_test_spectrum.py

Script to generate a spectrum for testing PySpecTools.

This uses the `pyspectools.fast.gaussian` to generate Gaussian
peaks at uniform random locations. The amplitudes are sampled
from a normal distribution.

Every peak is grouped together using random integers; these
will be used later to test the `assignment` module mechanisms.
"""

from pyspectools.fast.lineshapes import gaussian
import numpy as np
import pandas as pd
import yaml


SEED = 42
NPEAKS = 200
NGROUPS = 10
RESOLUTION = 0.02
FREQ_RANGE = (8000.0, 19000.0)

# Set a specific random seed to get consistent results
_ = np.random.seed(SEED)

# Gaussian parameters
AMPS = np.random.normal(loc=5.0, scale=2.0, size=(NPEAKS))
CENTERS = np.random.uniform(
    low=min(FREQ_RANGE) + 50.0, high=max(FREQ_RANGE) - 50.0, size=(NPEAKS)
)
# 50 kHz sigma
WIDTH = 0.05
# This generates encodings for each of the peaks as fake assignments
ASSIGNMENTS = np.random.randint(low=0, high=NGROUPS, size=(NPEAKS))

X = np.arange(min(FREQ_RANGE), max(FREQ_RANGE), RESOLUTION)
Y = np.random.normal(loc=0.0, scale=0.2, size=(X.size))
Y = np.abs(Y)

# Loop over each of parameters, and generate a Gaussian
for A, x0 in zip(AMPS, CENTERS):
    Y += gaussian(X, A, x0, WIDTH)

# Save the spectrum for tests
spectral_df = pd.DataFrame({"Frequency": X, "Intensity": Y})
spectral_df.to_csv("test-spectrum.csv", index=False)

# Package up the parameters used to generate the spectrum
# for testing
param_dict = {
    "NPEAKS": NPEAKS,
    "RESOLUTION": RESOLUTION,
    "FREQ_RANGE": FREQ_RANGE,
    "CENTERS": np.round(CENTERS, 4).tolist(),
    "AMPS": np.round(AMPS, 4).tolist(),
    "ASSIGNMENTS": ASSIGNMENTS.tolist(),
    "WIDTH": WIDTH,
}
with open("spectrum-parameters.yml", "w+") as write_file:
    yaml.dump(param_dict, write_file)
