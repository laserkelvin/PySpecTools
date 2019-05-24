# PySpecTools

## A Python library for analysis of rotational spectroscopy and beyond


## Version History

From version >= 3 of this module, Python 3.7 is required to support the new
`dataclass` objects which will be the basis for assigning broadband spectra.

## Introduction

`PySpecTools` is a library written to help with analyzing rotational
spectroscopy data. The main sorts of functionality are:

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

The library is very much in development, and is gradually being cleaned
up/improved on as my coding ability gets better.

---

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
For now, development 

---

## To do

1. Reorganize the coding, and make it more PEP friendly!
2. Example Jupyter notebooks
3. Improve autofit routines

