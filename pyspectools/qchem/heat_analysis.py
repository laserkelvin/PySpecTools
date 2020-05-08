"""
heat_analysis.py

Routines for performing batch analysis on HEAT345(Q) calculations.
The scripts are hardcoded to follow the filestructure produced by
the heat345.py scripts, i.e.

top/
|---heat345.py
|---heat345.yaml
|---zmat.yaml
|---correlation/
    |---AUG-PCVXZ/
        |---calcY-correlation-AUG-PCVXZ.log

and so on.
"""

import os
from glob import glob
import logging
import numpy as np
import pandas as pd
from scipy.constants import Avogadro

# Additional modules written by yours truly
from pyspectools.qchem import parsers
from pyspectools.qchem import extrapolation as ep
from pyspectools import routines

"""Analysis functions

These functions were designed for use with the HEAT pandas dataframes
"""

def calc_enthalpy(dataframe, reactants, products):
    # Calculate the enthalpy of reaction between lists of reactants and
    # products. The value is returned in kJ/mole, which is calculated
    # based on the explicit value of hartree to joule from NIST
    E_prod = np.sum([dataframe.loc[ID]["Total"] for ID in products])
    E_reac = np.sum([dataframe.loc[ID]["Total"] for ID in reactants])
    return (E_prod - E_reac) * (Avogadro * 4.359744650e-21)


def calc_reaction(dataframe, reactants, products, reactions):
    # A basic function for building sequential reactions
    # Could be useful for PES building
    energy = calc_enthalpy(dataframe, reactants, products)
    react_name = " + ".join(reactants) + "-->" + " + ".join(products)
    print(react_name + ": " + str(energy) + " kJ/mol")
    reactions.append([react_name, energy])
    

def get_hot(dataframe, species):
    # Return the HEAT345(Q) energy for one molecule
    return dataframe.loc[species]["Total"]


def relative_energies(df, base, molecules):
    # Returns a dataframe with all the energetics relative to
    # a single molecule
    # bases is a list of reactants, while molecules are a nested
    # list
    energetics = list()
    names = list()
    if base not in molecules:
        molecules.append(base)
    for pair in molecules:
        # Loop over all the pairs and calculate the relative energy
        energetics.append(calc_enthalpy(df, base, pair))
        names.append(" + ".join(pair))
    new_df = pd.DataFrame(energetics, index=names, columns=["Relative energy (kJ/mol)"])
    new_df["Coordinate"] = np.arange(len(molecules))
    return new_df


"""
    Low-level functions

    These functions are called by the higher level wrappers
    for performing the analysis.
"""


def analysis_lookup(method):
    """ Returns the function associated with the string
        based name of a function.
    """
    if method == "ZPE":
        func = anal_zpe
    elif method == "MVD":
        func = anal_mvd
    elif method == "DBOC":
        func = anal_dboc
    elif method == "CCSD(T)":
        func = anal_correlation
    elif method == "HLC":
        func = anal_hlc
    elif method == "(Q)":
        func = anal_q
    return func


def anal_zpe(data_dict):
    """ Function for parsing the ZPE out of the raw data dictionary.
        Takes the full dictionary as input, and returns the ZPE.
    """
    zpe = 0.
    # Different syntax for different CFOUR function calls
    for word in ["frequency", "freq"]:
        try:
            zpe = data_dict[word]["zpe"] / (4.12 * 627.509)
        except KeyError:
            pass
    return zpe


def anal_mvd(data_dict):
    """ Function for parsing the MVD1 + MVD2 out of the raw data dictionary.
        Takes the full dictionary as input, and returns the relativistic
        contributions to the energy.
    """
    return data_dict["rel"]["relativistic"]


def anal_dboc(data_dict):
    """ Returns the DBOC value """
    return data_dict["dboc"]["dboc"]


def anal_q(data_dict):
    """ Try and get the CCSDT(Q) energy """
    try:
        q_dict = data_dict["ccsdtq"]
        q_energy = q_dict["ccsdt(q) energy"] - q_dict["ccsdt energy"]
    except KeyError:
        q_energy = 0.
    return q_energy


def anal_correlation(data_dict):
    """ Function for extrapolating the correlation and SCF energies to
        the CBS limit using their respective schemes.
        Takes the full dictionary as input, and returns the SCF/CBS and
        CCSD(T)/CBS energies as floats
    """
    # Get the basis names from dictionary keys
    corr_basis = [name for name in data_dict.keys() if "correlation" in name]
    cardinals = list()
    bases = list()
    corr_energies = list()
    scf_energies = list()
    # This loop is deliberately written this way so that the ordering
    # is preserved
    for basis in corr_basis:
        basis_str = basis.split("-")[-1]
        basis_X = match_basis(basis_str)
        corr_energies.append(data_dict[basis]["ccsd(t) energy"])
        scf_energies.append(data_dict[basis]["scf energy"])
        cardinals.append(basis_X)
        bases.append(basis_str)
    # Package into pandas df
    extrap_df = pd.DataFrame(
            data=list(zip(bases, cardinals, scf_energies, corr_energies)),
            columns=["Basis", "Cardinal", "SCF energy", "CCSD(T) energy"]
            )
    # Not necessary, but makes things nicer
    extrap_df.sort_values(["Cardinal"], ascending=True, inplace=True)
    # Extrapolate SCF to CBS limit
    scf_cbs, scf_popt = ep.extrapolate_SCF(
            extrap_df["SCF energy"],
            extrap_df["Cardinal"]
            )
    # Extrapolate CCSD(T) correlation to CBS limit
    cc_cbs, cc_popt = ep.extrapolate_correlation(
            extrap_df["CCSD(T) energy"],
            extrap_df["Cardinal"]
            )
    return scf_cbs, cc_cbs, extrap_df


def anal_hlc(data_dict):
    """ Function for analyzing the non-perturbative corrections to
        the triple excitations.

        Takes the frozen-core CCSDT and CCSD(T) values at two basis
        and extrapolates them to the CBS limit.

        The end result is the difference between CCSDT/CBS and CCSD(T)/CBS
        
        This function returns the correction, as well as the dataframes
        holding the answers
    """
    # Get HLC perturbative terms
    pert_basis = [key for key in data_dict if "hlc-pert" in key]
    cardinals = list()
    pert_energies = list()
    full_energies = list()
    for basis in pert_basis:
        basis_str = basis.split("-")[-1]
        basis_X = match_basis(basis_str)
        cardinals.append(basis_X)
        scf = data_dict[basis]["scf energy"]
        corr = data_dict[basis]["ccsd(t) energy"]
        pert_energies.append(corr - scf)
    # Package the perturbative terms and extrapolate
    pert_df = pd.DataFrame(
            data=list(zip(cardinals, pert_energies)),
            columns=["Cardinal", "CCSD(T) energy"]
            )
    pert_cbs, pert_popt = ep.extrapolate_correlation(
            pert_df["CCSD(T) energy"],
            pert_df["Cardinal"]
            )
    # Get HLC non-perturbative terms
    pert_basis = [key for key in data_dict if "hlc-full" in key]
    cardinals = list()
    pert_energies = list()
    full_energies = list()
    for basis in pert_basis:
        basis_str = basis.split("-")[-1]
        basis_X = match_basis(basis_str)
        cardinals.append(basis_X)
        corr = data_dict[basis]["ccsd(t) energy"]
        pert_energies.append(corr)
    # Package the non-perturbative terms and extrapolate
    nonpert_df = pd.DataFrame(
            data=list(zip(cardinals, pert_energies)),
            columns=["Cardinal", "CCSDT energy"]
            )
    nonpert_cbs, nonpert_popt = ep.extrapolate_correlation(
            nonpert_df["CCSDT energy"],
            nonpert_df["Cardinal"]
            )
    correction = nonpert_cbs - pert_cbs
    return correction, pert_df, nonpert_df


def heat_analysis(mol_name, data_dict, methods=None, logger=None):
    """ Main driver function for analyzing HEAT contributions.

        This version has been written to be completely modular with
        respect to the contributions that it can take.

        Loops through a list of analysis functions and spits out
        a dataframe containing all of the resulting analysis.
    """
    if os.path.isdir("outputs") is False:
        os.mkdir("outputs")
    if methods is None:
        # If not specified, just default to the bare minimum 
        methods = [
                "ZPE",
                "CCSD(T)",
                ]
    results = dict()
    heat_energy = 0.
    for method in methods:
        # Get the function from string
        anal_func = analysis_lookup(method)
        # If the contribution is just a single value, add
        # straight to the dictionary
        if method in ["(Q)", "DBOC", "ZPE", "MVD"]:
            results[method] = anal_func(data_dict)
            if logger:
                logger.info(method + str(results[method]))
        elif method == "CCSD(T)":
            # Deal with the correlation
            scf_cbs, cc_cbs, extrap_df = anal_func(data_dict)
            results["SCF/CBS"] = scf_cbs
            results["CCSD(T)/CBS"] = cc_cbs
            extrap_df.to_csv("outputs/" + mol_name + "-SCF-CC.csv")
            if logger:
                logger.info("SCF/CC data")
                logger.info(extrap_df)
        elif method == "HLC":
            hlc, pert_df, nonpert_df = anal_func(data_dict)
            results["HLC"] = hlc
            pert_df.to_csv("outputs/" + mol_name + "-(T).csv")
            nonpert_df.to_csv("outputs/" + mol_name + "-T.csv")
            if logger:
                logger.info("Perturbative triple excitations")
                logger.info(pert_df)
                logger.info("Non-perturbative triple excitations")
                logger.info(nonpert_df)
    # Sum up all of the contributions
    for key, value in results.items():
        heat_energy+=value
    results["Total"] = heat_energy
    if logger:
        logger.info("Final energy: " + str(heat_energy))
    heat_df = pd.DataFrame.from_dict([results], orient="columns")
    heat_df.index = [mol_name]
    heat_df.to_csv("heat_analysis/" + mol_name + "-HEAT.csv")
    return heat_df


def read_heat(dir_path):
    """
    Function to parse in all of the calculations that consist of
    the HEAT scheme.
    Args: dir_path, path to top directory of the HEAT calculation
    which would end with the calcID

    Returns a dictionary containing all of the parsed logfiles
    """
    # Folder for keeping all the analysis logging outputs
    for folder in ["logs", "heat_analysis", "yml"]:
        if os.path.isdir(folder) is False:
            os.mkdir(folder)
    calcID = dir_path.split("/")[-1]
    logger = init_logger(calcID + "-HEAT.log")
    logger.info("HEAT345(Q) analysis of " + calcID)
    # Get the calculation folders
    dir_contents = glob(dir_path + "/*")
    dir_contents = [name for name in dir_contents
                    if os.path.isdir(name) is True]
    logger.info("Present folders: ")
    logger.info(dir_contents)
    results = dict()
    # Main loop over the calculation types
    # Perhaps there is a better way to code this up without explicit
    # hardcoding of if-cases, but for now this is how it'll be done.
    for calctype in dir_contents:
        calc_name = calctype.split("/")[-1]
        logger.info("Working on " + calctype)
        # For all cases other than the correlation, we have only one
        # calculation log file
        if "correlation" not in calctype and "hlc" not in calctype:
            try:
                logname = glob(calctype + "/*.log")[0]
                logger.info("Parsing " + logname)
                results[calc_name] = parsers.parse_cfour(logname)
            except IndexError:
                logger.info("Trouble parsing " + calctype)
        else:
            # There are multiple bases being used for correlation
            # and the HLC terms
            calc_list = glob(calctype + "/*/*.log")
            logger.info("Basis found: ")
            logger.info(calc_list)
            for basis in calc_list:
                # Get basis name from folder
                name = basis.split("/")[-2]
                results[calc_name + "-" + name] = parsers.parse_cfour(basis)
    logger.info("Done reading!")
    logger.info("Dumping results to " + calcID + "-parsed.yml")
    routines.dump_yaml("yml/" + calcID + "-parsed.yml", results)
    return results, logger


def analyze_molecule(mol_name, directory, methods=None):
    """ Automate analysis of a molecule by pointing to a directory
        containing all of the relevant calculations, as well as providing
        an identifier.
        
        Optional argument is to specify what contributions to include
        in the calculation

        Returns a dataframe containing the energy breakdown.
    """
    data_dict, logger = read_heat(directory)
    mol_df = heat_analysis(mol_name, data_dict, methods, logger)
    return mol_df


"""
    Miscellaneous functions

    Logging and basis matching
"""

def init_logger(log_name):
    """
    Use `logging` module to provide debug data for HEAT analysis.
    All of the extraneous data (e.g. extraplation coefficients, etc)
    will be output with the logger.

    Required input is the name of the log file including file extension.
    Returns a logger object.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("logs/" + log_name)
    fh.setLevel(logging.DEBUG)

    # Set up the formatter
    formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def match_basis(basis):
    """
    Converts a string representations of basis angular momenta
    to the cardinal number
    """
    # Remove the supplemental functions from the string as they don't
    # refer to the cardinal number
    basis = basis.replace("AUG-PC", "")
    basis = basis.replace("P", "")
    basis = basis.replace("C", "")
    basis_dict = {
            "VDZ": 2.,
            "VTZ": 3.,
            "VQZ": 4.,
            "V5Z": 5.,
            "V6Z": 6.
            }
    return basis_dict[basis]


def check_analysis(analysis_dict, logger):
    """
    Function for performing small sanity checks on the analysis results.
    Basic idea is to do some value comparisons to make sure the extrapolations
    have been performed sensibily.
    Args: dictionary containing the analysis values and a reference to the
    logger object used for debugging.
    The function will only print bad flags.
    """
    if np.abs(analysis_dict["CCSD(T)/CBS"] - analysis_dict["HLC-(T)"]) > 0.4:
        logger.warning("CCSD(T)/CBS and HLC-(T) difference > 0.4 Ha")
    if np.abs(analysis_dict["(Q)"] - analysis_dict["T - (T)"]) > 0.01:
        logger.warning("Large difference in CCSDT(Q) and T - (T)")
    if (analysis_dict["SCF/CBS"] / analysis_dict["HEAT345(Q)"]) < 0.99:
        logger.warning("Large contribution of HEAT345(Q) to SCF/CBS")
    #if np.abs(analysis_dict["HLC-T"] - analysis_dict["HLC-(T)"]) > 0.01:
    #    logger.warning("Large difference between HLC terms")
    if analysis_dict["ZPE"] == 0.:
        logger.warning("No ZPE - better be an atom!")
    if np.abs(analysis_dict["(Q)"]) > 1.:
        logger.warning("CCSD(T) contribution is too large to be true, setting to 0.")
        analysis_dict["(Q)"] = 0.

