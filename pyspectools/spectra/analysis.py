
from itertools import combinations, chain, compress
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Union
import warnings

import numpy as np
import pandas as pd
import peakutils
from astropy import units as u
from astroquery.splatalogue import Splatalogue
from lmfit import models, MinimizerException
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_samples
from scipy.signal import windows
from scipy.fftpack import rfft, irfft
from uncertainties import ufloat
from bokeh.layouts import layout
from bokeh.io import save
from plotly import graph_objs as go
import numba

from pyspectools import fitting
from pyspectools import lineshapes
from pyspectools import routines
from pyspectools import ftmw_analysis as fa
from pyspectools import figurefactory


def fit_line_profile(spec_df: pd.DataFrame, center: float, width=None, 
                     intensity=None, freq_col="Frequency", int_col="Intensity",
                     fit_func=models.GaussianModel, sigma=2, logger=None):
    """ 
        Somewhat high level function that wraps lmfit for
        fitting Gaussian lineshapes to a spectrum.

        For a given guess center and optional intensity,
        the 
    """
    model = fit_func()
    params = model.make_params()
    # Set up boundary conditions for the fit
    params["center"].set(
        center,
        min=center * 0.9997,
        max=center * 1.0003
    )
    if logger:
        logger.debug("Guess center: {:,.4f}.".format(center))
    if intensity:
        params["height"].set(
            intensity,
            min=0.
            )
    if width:
        params["sigma"].set(
            width,
            min=width * 0.95,
            max=width * 1.05
            )
    # Slice up a small chunk in frequency space; 0.5% of the
    # center frequency to allow for broad lineshapes
    freq_range = [center * offset for offset in [0.9995, 1.0005]]
    slice_df = spec_df[
        (spec_df[freq_col] >= freq_range[0]) & (spec_df[freq_col] <= freq_range[1])
    ]
    # Fit the peak lineshape
    fit_results = model.fit(
        data=slice_df[int_col],
        params=params,
        x=slice_df[freq_col],
    )
    if logger:
        logger.debug(fit_results.fit_report())
    if fit_results.success is True:
        names = list(fit_results.best_values.keys())
        fit_values = np.array([fit_results.best_values[key] for key in names])
        # Estimate standard error based on covariance matrix
        try:
            variance = np.sqrt(np.diag(fit_results.covar))
        except ValueError:
            variance = np.zeros(len(fit_values))
        percentage = (variance / fit_values) * 100.
        # In the instance where standard errors are large, we need to work out confidence intervals explicitly
        if len(percentage[percentage >= 5.]) > 0:
            # If using a Gaussian line profile, we can do extra statistics
            if logger:
                logger.warning("Large covariance detected; working out confidence intervals.")
            try:
                ci = fit_results.conf_interval()
                # Get the index corresponding to the right amount of sigma. Indices run 0 - 3 for one side, giving
                # 3, 2, 1 and 0 sigma values
                uncer = fit_values - np.array([ci[key][sigma - 1][1] for key in names])
            except (MinimizerException, ValueError):
                # Instances where changing a parameter does not affect the residuals at all; i.e. the cost function
                # is too flat w.r.t. to a parameter. In these cases we can't evaluate confidence intervals, and instead
                # we'll simply use the standard error of mean
                if logger:
                    logger.warning("Confidence intervals could not be evaluated, defaulting to standard error of mean.")
                uncer = variance * sigma
        else:
            # Otherwise, use the standard error of the mean as the uncertainty, multiplied by
            # the number of "sigma"
            uncer = variance * sigma
        # This bit is just formatting: format into a paired list then flatten
        summary_dict = {
            name: ufloat(value, uncertainty) for value, uncertainty, name in zip(fit_values, uncer, names)
        }
        summary_dict["Chi squared"] = fit_results.chisqr
        return fit_results, summary_dict
    else:
        return None, None


def peak_find(spec_df: pd.DataFrame, freq_col="Frequency", int_col="Intensity",
              thres=0.015, min_dist=10):
    """ 
        Wrapper for peakutils applied to pandas dataframes. First finds
        the peak indices, which are then used to fit Gaussians to determine
        the center frequency for each peak.

        Parameters
        ----------
        spec_df: dataframe
            Pandas dataframe containing the spectrum information, with columns corresponding to frequency and intensity.
        freq_col: str, optional
            Name of the frequency column in `spec_df`
        int_col: str, optional
            Name of the intensity column in `spec_df`
        thres: float, optional
            Threshold for peak detection

        Returns
        -------
        peak_df
            Pandas dataframe containing the peaks frequency/intensity
    """
    peak_indices = peakutils.indexes(
        spec_df[int_col].to_numpy(),
        thres=thres,
        thres_abs=True,
        min_dist=min_dist
        )
    frequencies = peakutils.interpolate(
        x=spec_df[freq_col].to_numpy(),
        y=spec_df[int_col].to_numpy(),
        ind=peak_indices,
        width=11
        )
    # Get the peaks if we were just using indexes
    direct_df = spec_df.iloc[peak_indices]
    direct_df.reset_index(inplace=True)
    direct_freqs = direct_df[freq_col].values
    # Calculate the difference in fit vs. approximate peak
    # frequencies
    differences = np.abs(direct_df[freq_col] - frequencies)
    intensities = spec_df.iloc[peak_indices][int_col].values
    peak_df = pd.DataFrame(
        data=list(zip(frequencies, direct_freqs, intensities)),
        columns=["Frequency", "Peak Frequencies", int_col]
        )
    # Take the indexed frequencies if the fit exploded
    # and deviates significantly from the original prediction
    peak_df.update(
        direct_df.loc[differences >= 0.2]
        )
    # Use 1sigma as the detection threshold; remove everything else!
    peak_df = peak_df.loc[
        peak_df[int_col] >= thres
        ]
    peak_df.reset_index(drop=True, inplace=True)
    return peak_df


def search_molecule(species: str, freq_range=[0., 40e3], **kwargs):
    """
    Function to search Splatalogue for a specific molecule. Technically I'd prefer to
    download entries from CDMS instead, but this is probably the most straight
    forward way.

    The main use for this function is to verify line identifications - if a line is
    tentatively assigned to a U-line, then other transitions for the molecule that
    are stronger or comparatively strong should be visible.

    Parameters
    ----------
    species: str
        Chemical name of the molecule
    freq_range: list
        The frequency range to perform the lookup

    Returns
    -------
    DataFrame or None
        Pandas dataframe containing transitions for the given molecule. If no matches are found, returns None.
    """
    default_param = {
        "line_lists": ["CDMS", "JPL"],
        "export_limit": 20000
    }
    default_param.update(**kwargs)
    splat_df = Splatalogue.query_lines(
        min(freq_range) * u.MHz,
        max(freq_range) * u.MHz,
        chemical_name=species,
        **default_param
    ).to_pandas()
    print(splat_df)
    if len(splat_df)> 0:
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
    else:
        return None


def search_center_frequency(frequency: float, width=0.5):
    """
    Function for wrapping the astroquery Splatalogue API for looking up a frequency and finding candidate molecules
    for assignment. The width parameter adjusts the +/- range to include in the search: for high frequency surveys,
    it's probably preferable to use a percentage to accommodate for the typically larger uncertainties (sub-mm
    experiments).

    Parameters
    ----------
    frequency: float
        Frequency in MHz to search Splatalogue for.
    width: float, optional
        Absolute frequency offset in MHz to include in the search.

    Returns
    -------
    dataframe or None
        Pandas dataframe containing frequency matches, or None if no matches are found.
    """
    min_freq = frequency - width
    max_freq = frequency + width
    try:
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
            "E_U (K)",
            "E_L (K)"
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
            "E_U (K)",
            "E_L (K)"
            ]
        # Now we combine the frequency measurements
        splat_df["Frequency"] = splat_df["Meas Freq-GHz"].values.astype(float)
        if splat_df["Frequency"].isna().any():
            try:
                # Replace missing experimental data with calculated
                splat_df["Frequency"].fillna(splat_df["Freq-GHz"], inplace=True)
            except ValueError as exception:
                warnings.warn(f"Problem with merging measured/predicted frequencies at {frequency:.4f}")
                warnings.warn(f"{exception}")
        # Convert to MHz
        splat_df["Frequency"] *= 1000.
        return splat_df
    except IndexError:
        warnings.warn("Could not parse Splatalogue table at {:,.4f}".format(frequency))
        return None


def calc_line_weighting(
        frequency: float, catalog_df: pd.DataFrame, prox=0.00005,
        abs=True, freq_col="Frequency", int_col="Intensity"
):
    """
        Function for calculating the weighting factor for determining
        the likely hood of an assignment. The weighting factor is
        determined by the proximity of the catalog frequency to the
        observed frequency, as well as the theoretical intensity if it
        is available.
        Parameters
        ----------------
         frequency : float
            Observed frequency in MHz
         catalog_df : dataframe
            Pandas dataframe containing the catalog data entries
         prox: float, optional
            Frequency proximity threshold
         abs: bool
            Specifies whether argument prox is taken as the absolute value
        Returns
        ---------------
        None
            If nothing matches the frequency, returns None.
        dataframe
            If matches are found, calculate the weights and return the
            candidates in a dataframe.
    """
    if abs is False:
        lower_freq = frequency * (1 - prox)
        upper_freq = frequency * (1 + prox)
    else:
        lower_freq = frequency - prox
        upper_freq = frequency + prox
    sliced_catalog = catalog_df.loc[
        catalog_df[freq_col].between(lower_freq, upper_freq)
    ]
    nentries = len(sliced_catalog)
    if nentries > 0:
        if int_col in sliced_catalog:
            column = sliced_catalog[int_col]
        elif "CDMS/JPL Intensity" in sliced_catalog:
            column = sliced_catalog["CDMS/JPL Intensity"]
        else:
            column = None
        # Vectorized function for calculating the line weighting
        sliced_catalog["Weighting"] = line_weighting(
            frequency, sliced_catalog[freq_col], column
        )
        # Normalize and sort the weights only if there are more than one
        # candidates
        if nentries > 1:
            sliced_catalog.loc[:, "Weighting"] /= sliced_catalog[
                "Weighting"].max()
            # Sort by obs-calc
            sliced_catalog.sort_values(["Weighting"], ascending=False,
                                       inplace=True)
        sliced_catalog.reset_index(drop=True, inplace=True)
        return sliced_catalog
    else:
        return None


def brute_harmonic_search(frequencies: np.ndarray, maxJ=10, dev_thres=5., prefilter=False):
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


def harmonic_finder(frequencies: np.ndarray, search=0.001, low_B=400., high_B=9000.):
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
        # Ignore everything that is too large or too small
        guess_B = np.diff(combo)
        if low_B <= guess_B <= high_B:
            combo = np.array(combo)
            # From B, determine the next series of lines and
            # find the closest ones
            candidates = find_series(combo, frequencies, search)          
            progressions.append(candidates)
    return progressions


def cluster_AP_analysis(progression_df: pd.DataFrame, sil_calc=False, refit=False, **kwargs):
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
    ap_options.update(**kwargs)
    ap_obj = AffinityPropagation(**ap_options)
    # Determine clusters based on the RMS, B, and D similarities
    # Remove occurrences of NaN in the three columns
    progression_df.dropna(subset=["RMS", "B", "D"], inplace=True)
    # Fit a random 40% subset of the B, D, results to the cluster model
    ap_obj.fit(progression_df.sample(frac=0.4)[["RMS", "B", "D"]])
    # Predict the clusters
    progression_df["Cluster indices"] = ap_obj.predict(
        progression_df[["RMS", "B", "D"]]
    )
    # Indicate which progression fits with which cluster
    #progression_df["Cluster indices"] = ap_obj.labels_
    # Calculate the metric for determining how well a sample
    # fits into its cluster
    if sil_calc is True:
        progression_df["Silhouette"] = silhouette_samples(
            progression_df[["RMS", "B", "D"]],
            progression_df["Cluster indices"],
            metric="euclidean"
        )
    data = dict()
    # Loop over each cluster, and aggregate the frequencies together
    for index, label in enumerate(progression_df["Cluster indices"].unique()):
        data[label] = dict()
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
        data[label]["Frequencies"] = unique_frequencies
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
            data[label].update(fit.best_values)
            data[label]["oldRMS"] = cluster_data[0]
            data[label]["RMS"] = np.sqrt(np.average(np.square(fit.residual)))
        else:
            # Reuse old RMS
            fit_values = {
                "B": cluster_data[1],
                "D": cluster_data[2],
                "RMS": cluster_data[0]
                }
            data[label].update(fit_values)
    return data, progression_df, ap_obj


def find_series(combo: Tuple[float, float], frequencies: np.ndarray, search=0.005):
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
        nearest_neighbours = frequencies[
            (frequencies >= lower_guess) & (frequencies <= upper_guess)
            ]
        # If we don't find anything close enough, don't worry about it
        # this will make sure that missing lines aren't necessarily ignored
        if len(nearest_neighbours) > 0:
            # Return the closest value to the predicted center
            candidates.append(
                nearest_neighbours[
                    np.argmin(np.abs(guess - nearest_neighbours))
                ]
            )
    return candidates


def create_cluster_tests(cluster_dict: Dict[Any, Any], shots=25, 
                         dipole=1.0, min_dist=500., **kwargs):
    """
    Take the output of the cluster AP analysis, and generate the FTB batch
    files for a targeted DR search.

    Parameters
    ----------
    cluster_dict: dict
        Cluster dictionary with keys corresponding to the cluster number,
        and values are subdictionaries holding the frequencies associated
        with the cluster.
    shots: int, optional
        Number of integration counts
    dipole: float, optional
        Approximate dipole moment to target
    min_dist: float, optional
        Minimum frequency difference between the cavity and DR frequencies.
    kwargs
        Additional kwargs are passed to the ftb line generation.

    """
    # In the event one wants to perform all of the clusters in a sitting
    full_str = ""
    # Loop over each cluster, and we'll generate an FTB file for that cluster
    for cluster, subdict in cluster_dict.items():
        filepath = f"ftb/cluster{cluster}-dr.ftb"
        frequencies = subdict["Frequencies"]
        ftb_str = ""
        ftb_dict = {"dipole": dipole}
        ftb_dict.update(**kwargs)
        for cavity_freq in frequencies:
            # First time a cavity frequency is used, tune there and turn
            # the DR off
            ftb_dict["drpower"] = -20
            ftb_str += fa.generate_ftb_line(
                cavity_freq,
                shots=shots,
                **ftb_dict
            )
            # Loop over the other frequencies
            for dr_freq in frequencies:
                # If we're sufficiently far away, then ping that transition
                if np.abs(cavity_freq - dr_freq) > min_dist:
                    ftb_dict["drpower"] = 13
                    ftb_dict["drfreq"] = "{:.4f}".format(dr_freq)
                    ftb_dict["skiptune"] = True
                    ftb_str += fa.generate_ftb_line(
                        cavity_freq,
                        shots=shots,
                        **ftb_dict
                    )
                else:
                    pass
        full_str += ftb_str
        with open(filepath, "w+") as write_file:
            write_file.write(ftb_str)
        with open("ftb/full-cluster-dr.ftb", "w+") as write_file:
            write_file.write(full_str)


def blank_spectrum(
        spectrum_df: pd.DataFrame, frequencies: np.ndarray, 
        noise=0., noise_std=0.05, freq_col="Frequency",
        int_col="Intensity", window=1., df=True):
    """
    Function to blank the peaks from a spectrum. Takes a iterable of frequencies, and generates an array of Gaussian
    noise corresponding to the average noise floor and standard deviation.

    Parameters
    ----------
    spectrum_df - pandas DataFrame
        Pandas DataFrame containing the spectral data
    frequencies - iterable of floats
        An iterable containing the center frequencies to blank
    noise - float
        Average noise value for the spectrum. Typically measured by choosing a region void of spectral lines.
    noise_std - float
        Standard deviation for the spectrum noise.
    freq_col - str
        Name of the column in spectrum_df to use for the frequency axis
    int_col - str
        Name of the column in spectrum_df to use for the intensity axis
    window - float
        Value to use for the range to blank. This region blanked corresponds to frequency+/-window.
    df - bool
        If True, returns a copy of the Pandas Dataframe with the blanked intensity.
        If False, returns a numpy 1D array corresponding to the blanked intensity.

    Returns
    -------
    new_spec - pandas DataFrame or numpy 1D array
        If df is True, Pandas DataFrame with the intensity regions blanked.
        If df is False, numpy 1D array
    """
    new_spec = spectrum_df.copy()
    # Reset the random number generator seed
    np.random.seed()
    for frequency in frequencies:
        # Work out the length of the noise window we have to create
        mask = new_spec[freq_col].between(
            frequency - window,
            frequency + window
        )
        length = len(new_spec.loc[mask])
        # Create Gaussian noise for this region
        noise_array = np.random.normal(noise, noise_std, length)
        new_spec.loc[mask, int_col] = noise_array
    if df is True:
        return new_spec
    else:
        return new_spec[int_col].values


def plotly_create_experiment_comparison(experiments, thres_prox=0.2, index=0,
                                       filepath=None, **kwargs):
    """
    Function to create a plot comparing multiple experiments. This is a high
    level function that wraps the `correlate_experiments` function,
    and provides a visual and interactive view of the spectra output from
    this function using Plotly.

    This function is effectively equivalent to
    `bokeh_create_experiment_comparison`, however uses Plotly as the front
    end instead.


    Parameters
    ----------
    experiments
    thres_prox
    index
    filepath
    kwargs

    Returns
    -------

    """
    fig = figurefactory.init_plotly_subplot(
        nrows=2, ncols=1,
        **{
            "subplot_titles":   ["Experiment Comparison", "Unique Spectrum"],
            "vertical_spacing": 0.15,
            "shared_xaxes":     True
        }
    )
    n_experiments = len(experiments)
    colors = figurefactory.generate_colors(
        n_experiments,
        "Set1",
        hex=True,
    )
    base_exp = deepcopy(experiments[index])
    top_traces = list()
    top_traces.append(go.Scattergl(
            x=base_exp.data[base_exp.freq_col],
            y=base_exp.data[base_exp.int_col] / base_exp.session.baseline,
            opacity=0.9,
            line={"color": colors[index]},
            name=f"Ref; {base_exp.session.experiment}"
        )
    )
    # Loop over all the experiments and plot them up
    for idx, experiment in enumerate(experiments):
        if idx != index:
            top_traces.append(go.Scattergl(
                    x=experiment.data[experiment.freq_col],
                    y=experiment.data[
                        experiment.int_col] / experiment.session.baseline,
                    opacity=0.9,
                    line={"color": colors[idx]},
                    name=f"{experiment.session.experiment}"
                )
            )
    fig.add_traces(top_traces, [1] * len(experiments), [1] * len(experiments))
    base, indices = correlate_experiments(
        experiments, thres_prox=thres_prox, index=index
    )
    bottom_trace = go.Scattergl(
        x=base.data["Frequency"],
        y=base.data["Unique Spectrum"] / base.session.baseline,
        opacity=0.9,
        line={"color": colors[index]},
        name="Unique Spectrum"
    )
    fig.add_traces([bottom_trace], [2], [1])
    fig["layout"].update(
        autosize=True,
        height=1000,
        width=900,
        showlegend=False
    )
    fig["layout"]["xaxis1"].update(
        title="Frequency (MHz)",
        showgrid=True,
        tickformat=":.0f"
    )
    for axis in ["yaxis1", "yaxis2"]:
        fig["layout"][axis].update(
            title="Signal-to-Noise",
            showgrid=True,
            tickformat=":.0f"
        )
    return fig


def bokeh_create_experiment_comparison(experiments, thres_prox=0.2, index=0,
                                 filepath=None, **kwargs):
    """
    Function to create a plot comparing multiple experiments. This is a high
    level function that wraps the `correlate_experiments` function,
    and provides a visual and interactive view of the spectra output from
    this function using Bokeh.


    Parameters
    ----------
    experiments
    thres_prox
    index
    filepath
    kwargs

    Returns
    -------

    """
    params = {
        "plot_height": 500,
        "plot_width": 900,
        "title": "Experiment Comparison"
    }
    params.update(**kwargs)
    full_fig = figurefactory.init_bokeh_figure(**params)
    full_fig.yaxis.axis_label = "Signal-to-noise"
    n_experiments = len(experiments)
    colors = figurefactory.generate_colors(
        n_experiments,
        "Pastel1",
        hex=True,
    )
    base_exp = deepcopy(experiments[index])
    full_fig.line(
        base_exp.data[base_exp.freq_col],
        base_exp.data[base_exp.int_col] / base_exp.session.baseline,
        line_width=2,
        alpha=0.9,
        color=colors[0],
        legend=f"Ref; {base_exp.session.experiment}"
    )
    # Loop over all the experiments and plot them up
    for idx, experiment in enumerate(experiments):
        if idx != index:
            full_fig.line(
                experiment.data[experiment.freq_col],
                experiment.data[experiment.int_col] / experiment.session.baseline,
                line_width=2,
                alpha=0.9,
                color=colors[idx],
                legend=f"{experiment.session.experiment}"
            )
    # Set some legend stuffs
    full_fig.legend.click_policy = "hide"
    full_fig.legend.location = "top_left"
    params = {
        "plot_height": 500,
        "plot_width": 900,
        "title": "Unique Features"
    }
    params.update(**kwargs)
    unique_fig = figurefactory.init_bokeh_figure(**params)
    base, indices = correlate_experiments(
        experiments, thres_prox=thres_prox, index=index
    )
    unique_fig.line(
        base.data["Frequency"],
        base.data["Unique Spectrum"] / base.session.baseline,
        line_width=2,
        alpha=0.9,
        color=colors[0]
    )
    unique_fig.yaxis.axis_label = "Signal-to-noise"
    combined_fig = layout([full_fig, unique_fig], sizing_mode="stretch_both")
    if filepath:
        save(combined_fig, filepath, title="ExperimentComparison")
        return None
    else:
        return combined_fig


def copy_assignments(A, B, corr_mat):
    """
    Function to copy assignments from experiment B over to experiment A. The
    correlation matrix argument requires the output from the
    `correlate_experiments` function.

    Parameters
    ----------
    A, B : AssignmentSession
        AssignmentSession objects, where the assignments from B are copied
        into A
    corr_mat : 2D array
        2D array mask with length A x B
    """
    # Find actual correlations; should end up having two 1D arrays of equal
    # length
    non_zero = np.where(corr_mat != 0)
    # Indexes of i and j correspond to A and B indexes
    for i, j in zip(non_zero[0], non_zero[1]):
        a_trans = A.line_lists["Peaks"].transitions[i]
        b_trans = B.line_lists["Peaks"].transitions[j]
        b_dict = b_trans.__dict__
        _ = b_dict.pop("frequency")
        a_trans.update(**b_dict)


def match_artifacts(on_exp, off_exp, thres=0.05, freq_col="Frequency"):
    """
    Function to remove a set of artifacts found in a blank spectrum.

    Parameters
    ----------
    on_exp - AssignmentSession object
        Experiment with the sample on; i.e. contains molecular features
    off_exp - AssignmentSession object
        Experiment with no sample; i.e. only artifacts
    thres - float, optional
        Threshold in absolute frequency units to match
    freq_col - str, optional
        Column specifying frequency in the pandas dataframes

    Returns
    -------
    candidates - dict
        Dictionary with keys corresponding to the uline index, and
        values the frequency
    """
    # check to make sure peaks are found
    for obj in [on_exp, off_exp]:
        if hasattr(obj, "peaks") is False:
            raise Exception("{} has no peaks!".format(obj.__name__))

    ufreqs = np.array([uline.frequency for index, uline in on_exp.ulines.items()])
    candidates = dict()
    for _, row in off_exp.peaks.iterrows():
        min_freq = row[freq_col] - thres
        max_freq = row[freq_col] + thres
        value, index = routines.find_nearest(ufreqs, row[freq_col])
        if min_freq <= value <= max_freq:
            candidates[index] = value
    return candidates


def line_weighting(frequency: float, catalog_frequency: float, intensity=None):
    """
    Function for calculating the line weighting associated with each assignment candidate. The formula is based on
    intensity and frequency offset, such as to favor strong lines that are spot on over weak lines that are further
    away.

    Parameters
    ----------
    frequency: float
        Center frequency in MHz; typically the u-line frequency.
    catalog_frequency: float
        Catalog frequency of the candidate
    intensity: float, optional
        log Intensity of the transition; includes the line strength and the temperature factor.

    Returns
    -------
    weighting: float
        Associated weight value. Requires normalization
    """
    deviation = np.abs(frequency - catalog_frequency)
    weighting = np.reciprocal(deviation)
    if intensity is not None:
        weighting *= 10.**intensity
    return weighting


def filter_spectrum(intensity: float, window: Union[str, np.ndarray] = "hanning", sigma=0.5):
    """
    Apply a specified window function to a signal. The window functions are
    taken from the `signal.windows` module of SciPy, so check what is available
    before throwing it into this function.

    The window function is convolved with the signal by taking the time-
    domain product, and doing the inverse FFT to get the convolved spectrum
    back.

    The one exception is the gaussian window - if a user specifies "gaussian"
    for the window function, the actual window function applied here is a
    half gaussian, i.e. a 1D gaussian blur.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Pandas dataframe containing the spectral information
    int_col: str, optional
        Column name to reference the signal
    window: str, optional
        Name of the window function as implemented in SciPy.
    sigma:

    Returns
    -------
    new_y: array_like
        Numpy 1D array containing the convolved signal
    """
    if isinstance(window, str):
        if window not in dir(windows):
            raise Exception("Specified window not available in SciPy.")
        data_length = len(intensity)
        if window == "gaussian":
            x = np.arange(data_length)
            window = lineshapes.gaussian(
                x, 1., 0., sigma
            )
        else:
            window = windows.get_window(window, data_length)
    elif isinstance(window, np.ndarray):
        pass
    else:
        raise Exception("Specified window not recognized.")
    # use Numpy function for cross-correlation
    new_y = np.correlate(intensity, window, mode="same")
    return new_y


def detect_artifacts(frequencies: np.ndarray, tol=2e-3):
    """
    Quick one-liner function to perform a very rudimentary test for
    RFI. This method relies on the assumption that any frequency that
    is suspiciously close to an exact number (e.g. 16250.0000) is very
    likely an artifact.
    
    The function will calculate the difference between each frequency
    and its nearest whole number, and return frequencies that are
    within a specified tolerance.
    
    Parameters
    ----------
    frequencies : NumPy 1D array
        Array of frequencies to check for artifacts.
    tol : float, optional
        Maximum tolerance to be used to check whether frequency
        is close enough to its rounded value, by default 2e-3
    
    Returns
    -------
    NumPy 1D array
        Returns the frequencies that match the specified criteria.
    """
    mask = np.abs(np.round(frequencies) - frequencies) <= tol
    return frequencies[mask]


@numba.njit(fastmath=True)
def cross_correlate(a: np.ndarray, b: np.ndarray, lags=None):
    """
    Cross-correlate two arrays a and b that are of equal length by
    lagging b with respect to a. Uses np.roll to shift b by values
    of lag, and appropriately zeros out "out of bounds" values.
    
    Parameters
    ----------
    a, b: Length n NumPy 1D arrays
        Arrays containing the values to cross-correlate. Must be the
        same length.
    lags : [type], optional
        [description], by default None
    """
    if lags is None:
        lags = np.arange(-50, 50, 1, dtype=int)
    assert lags.dtype == int
    assert a.size == b.size
    C = np.zeros(lags.size, dtype=a.dtype)
    index = 0
    for lag in lags:
        b_temp = np.roll(b, lag)
        if lag < 0:
            b_temp[b_temp.size + lag:] = 0
        elif lag > 0:
            b_temp[:lag] = 0
        else:
            pass
        C[index] = (a * b_temp).sum()
        index += 1
    return C, lags


def average_spectra(*arrays, **options) -> np.ndarray:
    """
    Averages multiple spectra together, with a few options available to
    the user. User provides a iterable of arrays, where the frequency
    axis of each of the arrays are the same, and the length of the arrays
    are also the same.
    
    Options include performing a noise-weighted average, where the spectra
    are averaged based on their inverse of their average baseline determined
    by an ALS fit. The lower the average noise, the larger weighting given
    to the spectrum.
    
    Parameters
    ----------
    arrays - np.ndarray
        Iterable of NumPy arrays corresponding to the intensity
        
    time_domain - bool
        If True, the averaging is done in the time domain, providing
        the input spectra are frequency domain. Defaults to False.
        
    weighted - bool
        If True, weights the averaging by the average noise in each
        spectrum. Defaults to True.    
    
    Returns
    -------
    np.ndarray
        Averaged intensities.
    
    Raises
    ------
    ValueError
        Error is raised if fewer than two spectra are given.
    """
    n_spectra = len(arrays)
    if n_spectra <= 1:
        raise ValueError("Need at least two spectra to coaverage!")
    time_domain = options.pop("time_domain", False)
    weighted = options.pop("weighted", True)
    averaged_spectra = np.zeros(arrays[0].size)
    weighting = list()
    # Loop over the spectra
    for index, spectra in enumerate(arrays):
        y = spectra.copy()
        als_params = {"lam": 1e2, "p": 0.05, "niter": 10}
        if time_domain is True:
            y = rfft(y)
        if weighted is True:
            baseline = fitting.baseline_als(y, **als_params)
            # Inverse of noise is used for weighting
            noise = 1 / np.average(baseline)
            weighting.append(noise)
            y *= noise
        averaged_spectra += y
    if weighted is True:
        # Divide through by the sum of weights
        norm = np.sum(weighting)
    else:
        # Otherwise, simple arthimetic mean
        norm = n_spectra
    averaged_spectra /= norm
    if time_domain is True:
        averaged_spectra = irfft(averaged_spectra)            
    return averaged_spectra