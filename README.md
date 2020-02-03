# PySpecTools

## A Python library for analysis of rotational spectroscopy and beyond

---

## Introduction

![pst-logo](docs/source/pst_logo_landscape.png)

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

The documentation for PySpecTools can be found [here](https://laserkelvin.github.io/PySpecTools).

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
a new version with potentially backwards breaking changes are about to be made.

## Planned features

1. Integration of deep learning tools for molecule identifiction and spectral assignment
2. Probability-based assignment routines - rather than single assignments.
3. Revamp of codebase - needs a substantial re-organization that will likely result in backwards compatibility breaking.
4. Additional Cython routines - many functions within `PySpecTools` are fast enough, but we can always go faster 😀

---

## Questions? Comments?

If you have features you would like to have added, please raise an issue on the repo, or
feel free to send me an email at kinlee_at_cfa.harvard.edu.

Also, please feel free to fork and contribute! The code is being formatted with `black`,
and uses NumPy-style docstrings. If you have any questions about contributing, drop me an
email!
