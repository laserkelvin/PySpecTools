
from astroquery.splatalogue import Splatalogue
from astropy import units as u
from lmfit import models
import numpy as np
import pandas as pd
import peakutils
from scipy.signal import savgol_filter
from . import plotting
from pyspectools import fitting
from itertools import combinations, chain
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_samples


def fit_line_profile(spec_df, frequency, intensity=None, name=None, verbose=False):
    """ Low level function wrapper to to fit line profiles
        in chirp spectra.
    """
    if "Cleaned" not in spec_df.columns:
        spec_df["Cleaned"] = spec_df["Intensity"].values
    model = models.VoigtModel()
    params = model.make_params()
    # Set up boundary conditions for the fit
    params["center"].set(
        frequency,
        min=frequency - 0.03,
        max=frequency + 0.03
    )
    # If an intensity is supplied
    if intensity:
        params["amplitude"].set(intensity)
    params["sigma"].set(
        0.05,
        min=0.04,
        max=0.07
    )
    freq_range = [frequency + offset for offset in [-5., 5.]]
    slice_df = spec_df[
        (spec_df["Frequency"] >= freq_range[0]) & (spec_df["Frequency"] <= freq_range[1])
    ]
    # Fit the peak lineshape
    fit_results = model.fit(
        slice_df["Intensity"],
        params,
        x=slice_df["Frequency"],
    )
    if verbose is True:
        print(fit_results.fit_report())
    yfit = fit_results.eval(x=spec_df["Frequency"])
    if name:
        spec_df[name] = yfit
    # Subtract the peak contribution
    spec_df["Cleaned"] -= yfit
    return fit_results


def peak_find(spec_df, col="Intensity", thres=0.015):
    """ Wrapper for peakutils applied to pandas dataframes """
    peak_indices = peakutils.indexes(
        spec_df[col],
        thres=thres,
        min_dist=10
        )
    return spec_df.iloc[peak_indices]


def search_center_frequency(frequency, width=0.5):
    """ Wrapper for the astroquery Splatalogue search
        This function will take a center frequency, and query splatalogue
        within the CDMS and JPL linelists for carriers of the line.

        Input arguments:
        frequency - float specifying the center frequency
    """
    min_freq = frequency - width
    max_freq = frequency + width
    splat_df = Splatalogue.query_lines(
        min_freq*u.MHz,
        max_freq*u.MHz,
        line_lists=["CDMS", "JPL"]
    ).to_pandas()
    # These are the columns wanted
    columns = [
        "Species",
        "Chemical Name",
        "Meas Freq-GHz(rest frame,redshifted)",
        "Freq-GHz(rest frame,redshifted)",
        "Resolved QNs",
        "CDMS/JPL Intensity",
        "E_U (K)"
        ]
    # Take only what we want
    splat_df = splat_df[columns]
    splat_df.columns = [
        "Species",
        "Chemical Name",
        "Meas Freq-GHz",
        "Freq-GHz",
        "Resolved QNs",
        "CDMS/JPL Intensity",
        "E_U (K)"
        ]
    # Now we combine the frequency measurements
    splat_df["Frequency"] = splat_df["Meas Freq-GHz"].values
    # Replace missing experimental data with calculated
    splat_df["Frequency"].fillna(splat_df["Freq-GHz"], inplace=True)
    # Convert to MHz
    splat_df["Frequency"] *= 1000.
    return splat_df


def assign_peaks(spec_df, frequencies, **kwargs):
    """ Higher level function that will help assign features
        in a chirp spectrum.

        Input arguments:
        spec_df - pandas dataframe containing chirp data
        frequencies - iterable containing center frequencies of peaks
        Optional arguments are passed into the peak detection as well
        as in the plotting functions.

        Returns a dataframe containing all the assignments, and a list
        of unidentified peak frequencies.
    """
    unassigned = list()
    dataframes = pd.DataFrame()
    for frequency in frequencies:
        splat_df = search_center_frequency(frequency)
        nitems = len(splat_df)
        # Only act if there's something found
        if nitems > 0:
            splat_df["Deviation"] = np.abs(
                    splat_df["Combined"] - frequency
                    )
            print(splat_df)
            try:
                print("Peak frequency is " + str(frequency))
                index = int(
                    input(
                        "Please choose an assignment index: 0-" + str(nitems - 1)
                        )
                )
                assigned = True
            except ValueError:
                print("Deferring assignment")
                unassigned.append(frequency)
                assigned = False
            # Take the assignment. Double brackets because otherwise
            # a series is returned rather than a dataframe
            if assigned is True:
                assignment = splat_df.iloc[[index]].sort_values(
                    ["Deviation"],
                    ascending=False
                )
                ass_freq = assignment["Combined"]
                # If the measurement is not available, go for
                # the predicted value
                ass_name = assignment["Species"] + "-" + assignment["Resolved QNs"]
                # Clean the line
                _ = fit_line_profile(spec_df, frequency)
                # Keep track of the assignments in a dataframe
                dataframes = dataframes.append(assignment)
            print("----------------------------------------------------")
        else:
            print("No species known for " + str(frequency))
    return dataframes, unassigned


def harmonic_search(frequencies, maxJ=10, dev_thres=5., prefilter=False):
    """
        Function that will search for possible harmonic candidates
        in a list of frequencies. Wraps the lower level function.

        Generates every possible 4 membered combination of the
        frequencies, and makes a first pass filtering out unreasonable
        combinations.

        parameters:
        ----------------
        frequencies - iterable containing floats of frequencies (ulines)
        maxJ - maximum value of J considered for quantum numbers
        dev_thres - standard deviation threshold for filtering unlikely
                    combinations of frequencies
        prefilter - bool dictating whether or not the frequency lists
                    are prescreened by standard deviation. This potentially
                    biases away from missing transitions!

        returns:
        ----------------
        results_df - pandas dataframe containing RMS information and fitted
                     constants
        fit_results - list containing all of ModelResult objects
    """
    frequencies = np.sort(frequencies)
    # List for holding candidates
    candidates = list()
    
    print("Generating possible frequency combinations.")
    # Sweep through all possible combinations, and look
    # for viable candidates
    if prefilter is True:
        for length in [3, 4, 5]:
            # Check the length of array we need...
            if comb(len(frequencies), length) > 5e6:
                pass
            combos = np.array(list(combinations(frequencies, length)))
            # Calculate the standard deviation between frequency
            # entries - if the series is harmonic, then the deviation
            # should be low and only due to CD terms
            deviation = np.std(np.abs(np.diff(combos, n=2)), axis=1)
            combos = combos[deviation < 100.]
            deviation = deviation[deviation < 100.]
            sorted_dev = np.sort(deviation)[:50]
            sorted_indexes = np.argsort(deviation)[:50]
            candidates.extend(list(combos[sorted_indexes]))
        print("Number of candidates: {}".format(len(candidates)))
    elif prefilter is False:
        # If we won't prefilter, then just chain the
        # generators together
        # THIS WILL BE FREAKING SLOW
        candidates = chain(
            combinations(frequencies, 3),
            combinations(frequencies, 4),
            combinations(frequencies, 5)
        )

    data_list = list()
    fit_results = list()
    if prefilter is True:
        progress = np.array([0.25, 0.50, 0.75])
        progress = progress * len(candidates)
        progress = [int(prog) for prog in progress]

    print("Looping over candidate combinations")
    # Perform the fitting procedure on candidate combinations
    for index, candidate in enumerate(candidates):
        # Only fit the ones that 
        min_rms, min_index, _, fit_values, fit_objs = fitting.harmonic_fit(
            candidate, 
            maxJ=maxJ, 
            verbose=False
            )
        data_list.append(
            [index, 
             min_rms / len(candidate), 
             candidate,
             *list(fit_values[min_index].values())]
            )
        fit_results.append(fit_objs[min_index])
        if prefilter is True:
            if index in progress:
                print("{} candidates screened.".format(index))

    print("Finalizing results.")
    results_df = pd.DataFrame(
        data=data_list,
        columns=["Index", "RMS", "Frequencies", "B", "D"]
        )

    results_df.sort_values(["RMS"], ascending=True, inplace=True)

    return results_df, fit_results


def cluster_AP_analysis(fit_df, damping=0.7, preference=-1e5, **kwargs):
    """
        Wrapper for the AffinityPropagation cluster method from
        scikit-learn.

        The dataframe provided will also receive new columns: Cluster index,
        and Silhouette. The latter corresponds to how likely a sample is
        sandwiched between clusters (0), how squarely it belongs in the
        assigned cluster (+1), or does not belong (-1). The cluster index
        corresponds to which cluster the sample belongs to.

        parameters:
        ---------------
        fit_df - pandas dataframe taken from the result of progression
                 fits
        damping - optional float larger than 0.5 and less than 1.0 for
                  minimizing numerical oscillation
        preference - float that corresponds to the number of exemplars/clusters.
                     Larger negative numbers correspond to fewer clusters.
        
        returns:
        --------------
        ap_obj - AffinityPropagation object containing all the information
                 as attributes.
    """
    ap_obj = AffinityPropagation(
        damping=damping,
        preference=preference,
        **kwargs)
    ap_obj.fit(fit_df[["RMS", "B", "D"]])

    # Labels/indices used to identify which set of
    # fits belong to which cluster
    fit_df["Cluster index"] = ab_obj.labels_
    
    # Calculate the Euclidean distance between
    # samples, the cluster it belongs to, and the
    # nearest clusters.
    fit_df["Silhouette"] = silhouette_samples(
        fit_df[["RMS", "B", "D"]],
        fit_df["Cluster index"],
        metric="euclidean"
        )

    return ap_obj
    
