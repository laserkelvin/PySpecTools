
from itertools import combinations, chain

import numpy as np
import pandas as pd
import peakutils
from astropy import units as u
from astroquery.splatalogue import Splatalogue
from lmfit import models
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_samples

from pyspectools import fitting


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


def peak_find(spec_df, freq_col="Frequency", int_col="Intensity", thres=0.015):
    """ 
        Wrapper for peakutils applied to pandas dataframes. First finds
        the peak indices, which are then used to fit Gaussians to determine
        the center frequency for each peak.

        parameters:
        ---------------
        spec_df - dataframe containing the spectrum
        freq_col - str denoting frequency column
        int_col - str denoting intensity column
        thres - threshold for peak detection

        returns:
        ---------------
        peak_df - pandas dataframe containing the peaks frequency/intensity
    """
    peak_indices = peakutils.indexes(
        spec_df[int_col],
        thres=thres,
        min_dist=10
        )
    frequencies = peakutils.interpolate(
        x=spec_df[freq_col].values,
        y=spec_df[int_col].values,
        ind=peak_indices,
        width=20
        )
    # Get the peaks if we were just using indexes
    direct_df = spec_df.iloc[peak_indices]
    direct_df.reset_index(inplace=True)
    # Calculate the difference in fit vs. approximate peak
    # frequencies
    differences = np.abs(direct_df[freq_col] - frequencies)
    intensities = spec_df.iloc[peak_indices][int_col].values
    peak_df = pd.DataFrame(
        data=list(zip(frequencies, intensities)),
        columns=["Frequency", "Intensity"]
        )
    # Take the indexed frequencies if the fit exploded
    # and deviates significantly from the original prediction
    peak_df.update(
        direct_df.loc[differences >= 0.5]
        )
    return peak_df


def search_molecule(species, freq_range=[0., 40e3]):
    """
    Function to search Splatalogue for a specific molecule. Technically I'd prefer to
    download entries from CDMS instead, but this is probably the most straight
    forward way.

    The main use for this function is to verify line identifications - if a line is
    tentatively assigned to a U-line, then other transitions for the molecule that
    are stronger or comparatively strong should be visible.
    :param species: str for the chemical name of the molecule
    :param freq_range: list for the frequency range considered
    :return: DataFrame containing transitions for the molecule
    """
    splat_df = Splatalogue.query_lines(
        min(freq_range) * u.MHz,
        max(freq_range) * u.MHz,
        chemical_name=species,
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


def brute_harmonic_search(frequencies, maxJ=10, dev_thres=5., prefilter=False):
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
            #if comb(len(frequencies), length) > 5e6:
            #    pass
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


def harmonic_finder(frequencies, search=0.001, low_B=400., high_B=9000.):
    """
        Function that will generate candidates for progressions.
        Every possible pair combination of frequencies are
        looped over, consider whether or not the B value is either
        too small (like C60 large) or too large (you won't have
        enough lines to make a progression), and search the
        frequencies to find the nearest candidates based on a
        prediction.
        
        parameters:
        ----------------
        frequencies - array or tuple-like containing the progressions
                      we expect to find
        search - optional argument threshold for determining if something
                 is close enough
                 
        returns:
        ----------------
        progressions - list of arrays corresponding to candidate progressions
    """
    frequencies = np.sort(frequencies)
    progressions = list()
    for combo in combinations(frequencies, 2):
        # Ignore everything that is too large or
        # too small
        guess_B = np.diff(combo)
        if low_B <= guess_B <= high_B:
            combo = np.array(combo)
            # From B, determine the next series of lines and
            # find the closest ones
            candidates = find_series(combo, frequencies, search)          
            progressions.append(candidates)
    return progressions


def cluster_AP_analysis(progression_df, sil_calc=False, refit=False, **kwargs):
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
        progression_df - pandas dataframe taken from the result of progression
                         fits
        sil_calc - bool indicating whether silhouettes are calculated
                   after the AP model is fit
        
        returns:
        --------------
        data - dict containing clustered frequencies and associated fits
        ap_obj - AffinityPropagation object containing all the information
                 as attributes.
    """
    ap_options = dict()
    ap_options.update(kwargs)
    print(ap_options)
    ap_obj = AffinityPropagation(**ap_options)
    # Determine clusters based on the RMS, B, and D
    # similarities
    print("Fitting the Affinity Propagation model.")
    ap_obj.fit(progression_df[["RMS", "B", "D"]])
    print("Fit complete.")
    progression_df["Cluster indices"] = ap_obj.labels_
    print("Determined {} clusters.".format(len(ap_obj.cluster_centers_)))
    # Calculate the metric for determining how well a sample
    # fits into its cluster
    if sil_calc is True:
        print("Calculating silhouettes.")
        progression_df["Silhouette"] = silhouette_samples(
            progression_df[["RMS", "B", "D"]],
            progression_df["Cluster indices"],
            metric="euclidean"
            )
    
    data = dict()
    print("Aggregating results.")
    for index, label in enumerate(progression_df["Cluster indices"].unique()):
        data[index] = dict()
        cluster_data = ap_obj.cluster_centers_[index]
        slice_df = progression_df.loc[progression_df["Cluster indices"] == label]
        columns = list()
        for col in progression_df:
            try:
                columns.append(int(col))
            except ValueError:
                pass
        unique_frequencies = np.unique(slice_df[columns].values.flatten())
        unique_frequencies = unique_frequencies[~np.isnan(unique_frequencies)]
        data[index]["Frequencies"] = unique_frequencies
        if refit is True:
            # Refit the whole list of frequencies with B and D again
            BJ_model = models.Model(fitting.calc_harmonic_transition)
            params = BJ_model.make_params()
            params["B"].set(
                cluster_data[1],
                min=cluster_data[1]*0.99,
                max=cluster_data[1]*1.01
                )
            params["D"].set(cluster_data[2], vary=False)
            # Get values of J based on B again
            J = (unique_frequencies / cluster_data[1]) / 2
            fit = BJ_model.fit(
                data=unique_frequencies,
                J=J,
                params=params
            )
            # Package results together
            fit_values = fit.best_values
            data[index].update(fit.best_values)
            data[index]["oldRMS"] = cluster_data[0]
            data[index]["RMS"] = np.sqrt(np.average(np.square(fit.residual)))
        else:
            # Reuse old RMS
            fit_values = {
                "B": cluster_data[1],
                "D": cluster_data[2],
                "RMS": cluster_data[0]
                }
            data[index].update(fit_values)
    return data, ap_obj


def find_series(combo, frequencies, search=0.005):
    """
        Function that will exhaustively search for candidate
        progressions based on a pair of frequencies.
        
        The difference of the pair is used to estimate B,
        which is then used to calculate J. These values of
        J are then used to predict the next set of lines,
        which are searched for in the soup of frequencies.
        The closest matches are added to a list which is returned.

        This is done so that even if frequencies are missing
        a series of lines can still be considered.

        parameters:
        ---------------
        combo - pair of frequencies corresponding to initial guess
        frequencies - array of frequencies to be searched
        search - optional threshold for determining the search range
                 to look for candidates

        returns:
        --------------
        array of candidate frequencies
    """
    lowest = np.min(combo)
    approx_B = np.average(np.diff(combo))
    minJ = (lowest / approx_B) / 2
    J = np.arange(minJ, minJ + 20., 0.5)
    # Guess where all the next frequencies are
    guess_centers = J * 2 * approx_B
    # Make sure it's within the band of trial frequencies
    guess_centers = guess_centers[guess_centers <= np.max(frequencies)]
    candidates = list()
    for guess in guess_centers:
        lower_guess = guess * (1 - search)
        upper_guess = guess * (1 + search)
        nearest_neighbours = frequencies[(frequencies >= lower_guess) & (frequencies <= upper_guess)]
        # If we don't find anything close enough, don't worry about it
        # this will make sure that missing lines aren't necessarily ignored
        if len(nearest_neighbours) > 0:
            # Return the closest value to the predicted center
            candidates.append(nearest_neighbours[np.argmin(np.abs(guess - nearest_neighbours))])
    return candidates
