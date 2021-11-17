# PySpecTools

## A Python library for analysis of rotational spectroscopy and beyond

---

## Introduction

[![DOI](https://zenodo.org/badge/90773952.svg)](https://zenodo.org/badge/latestdoi/90773952)
[![Build Status](https://travis-ci.com/laserkelvin/PySpecTools.svg?branch=master)](https://travis-ci.com/laserkelvin/PySpecTools)

![pst-logo](docs/source/_images/pst_logo_landscape.png)

`PySpecTools` is a library written to help with analyzing rotational
spectroscopy data. The main functions of this library are:

1. Wrapper for SPFIT and SPCAT programs of Herb Pickett, with YAML/JSON
   interpretation
2. Generating specific figure types using `matplotlib`, such as polyads and
   potential energy diagrams
3. Parsing and filtering of Fourier-transform centimeter-wave and
   millimeter-wave absorption data. This includes:
   - Fitting of lineshapes (e.g. Lorentizan second-derivative profiles)
   - Fourier-filtering
   - Double resonance fitting
4. Analysis of broad band spectra with the `AssignmentSession` and `Transition` classes.
   These classes, combined with Jupyter notebooks, provide a way to assign spectra
   reproducibly; astronomical and laboratory broadband spectra are supported.
5. Autofit routines are available for a set of special cases, like linear/prolate
   molecules. Eventually, SPFIT will be a backend option.
6. Molecule identity inference (NEW!)—this uses a pre-trained probabilistic deep
   learning model that allows users to perform inference on experimental constants
   and expected composition to predict the most likely molecular formula and possible
   functional groups present. See [our paper on the development of the first generation of this model](https://pubs.acs.org/doi/10.1021/acs.jpca.0c01376). An example of how to run this analysis
   can be found [here.](https://laserkelvin.github.io/PySpecTools/examples/identifying_molecules.html)

The documentation for PySpecTools can be found [here](https://laserkelvin.github.io/PySpecTools).

If you use PySpecTools for research, please cite use the DOI badge below to cite the version
of the package; this is not to track adoption, but rather for the sake of reproducibility!

## Installation

`conda` is the preferred way of maintaining software environments, and `poetry` is used to manage Python package dependencies.

As of PySpecTools 4.6.1, the installation is intended to be significantly more straightforward
with PyPI releases; in a given Python environment, just run:

`pip install PySpecTools`

Alternatively, if you're having issues, we recommend creating a new Python environment
within `conda`; with Python 3.7+:

1. `conda create -n pst python=3.7`
2. `conda activate pst`
3. `pip install poetry`
4. `poetry install`

Installation on Windows is less straightforward. The following instructions avoid
issues originating from virtual environments created by poetry and include a workaround
for a known issue with poetry in Windows.

1. `conda create -n pst python=3.7`
2. `conda activate pst`
3. `pip install poetry`
4. `poetry config virtualenvs.in-project false`
5. `poetry config virtualenvs.create false`
6. Navigate to the folder `C:\Users\user\AppData\Local\pypoetry\Cache` and delete all contents of this folder.
7. Navigate to the folder containing PySpecTools
8. `poetry install`

## PyPickett

`PySpecTools` includes a set of routines for wrapping SPFIT/SPCAT. The design
philosophy behind these functions is that the formatting and running of
SPFIT/SPCAT can be a little tricky, in terms of reproducibility, parameter
coding, and visualization. These problems are solved by wrapping and managing
input files in an object-oriented fashion:

1. Able to serialize SPFIT/SPCAT files from more human-friendly formats like
   YAML and JSON.
2. Automatic file/folder management, allowing the user to go back to an earlier
   fit/parameters. Ability to "finalize" the fit so the final parameter set is
   known.
3. Display the predicted spectrum using `matplotlib` in a Jupyter notebook,
   which could be useful for analysis and publication.
4. A parameter scan mode, allowing the RMS to be explored as a function of
   whatever parameter.

There is still much to do for this module, including a way of managing experimental lines.

## Notes on release

`PySpecTools` is currently being released on a irregular schedule, using a sequence-based version numbering system.
The numbering works as X.Y.Z, where X denotes huge changes that are backwards incompatible, Y are significant changes
(typically new features) and Z are minor bug fixes. A freeze and release will typically occur when
a new version with potentially backwards breaking changes are about to be made. The large changes typically occur once a year (based on the trend so far).

Currently, `PySpecTools` is under the MIT license, which allows anyone to freely use and modify as you wish!

## Planned features

1. Integration of deep learning tools for molecule identifiction and spectral assignment
2. Probability-based assignment routines - rather than single assignments.
3. Revamp of codebase - needs a substantial re-organization that will likely result in backwards compatibility breaking.
4. Additional Cython routines - many functions within `PySpecTools` are fast enough, but we can always go faster 😀
5. Better abstraction in the `spectra.assignment` modules - need to move a lot of the complicated routines into subclasses (especially for transitions and line lists), although there is a case to be made for a simpler user interface (only have to deal with `LineList`, instead of three subclasses of `LineList`)

## Contributing

If you have features you think would benefit other spectroscopists, you can raise an issue in the repo. Alternatively (and even better) would be to fork the repo, and submit a pull request!

The only comments on coding style are: 

1. Documentation is written in NumPy style
2. Object-oriented Python
3. Formatted with [`black`](https://black.readthedocs.io/en/stable/)

There are a set of unit tests that can be run to ensure the most complicated routines in the
library are working as intended. Right now coverage is poor, and so the tests I've written
focus on the `assignment` module. There is a script contained in the `tests` folder that will
generate a synthetic spectrum to test functionality on. To run these tests:

``` python
cd tests
python generate_test_spectrum.py
pytest
```

You will need to have `pytest` installed. These tests are designed to raise errors when there
are huge errors; some tolerance is included for imperfect peak detection, for example. These
are defined as constants within the `test_assignment.py` testing script.

---

## Questions? Comments?

If you have features you would like to have added, please raise an issue on the repo, or
feel free to send me an email at kinlee_at_cfa.harvard.edu.

Also, please feel free to fork and contribute! The code is being formatted with `black`,
and uses NumPy-style docstrings. If you have any questions about contributing, drop me an
email!
