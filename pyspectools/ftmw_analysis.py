from pyspectools import parsecat as pc
import pandas as pd
import numpy as np
import peakutils
from matplotlib import pyplot as plt
from matplotlib import colors
from plotly.offline import plot, init_notebook_mode
import plotly.graph_objs as go

def parse_specdata(filename):
    # For reading the output of a SPECData analysis
    return pd.read_csv(filename, skiprows=4)

def parse_spectrum(filename, threshold=20.):
    """ Function to read in a blackchirp or QtFTM spectrum from file """
    dataframe =  pd.read_csv(
        filename, delimiter="\t", names=["Frequency", "Intensity"], skiprows=1
    )
    return dataframe[dataframe["Intensity"] <= threshold]

def center_cavity(dataframe, thres=0.3, verbose=True):
    """ Finds the center frequency of a Doppler pair in cavity FTM measurements
        and provides a column of offset frequencies.

        Sometimes the peak finding threshold has to be tweaked to get the center
        frequency correctly.
    """
    # Find the peak intensities
    center_indexes = peakutils.indexes(dataframe["Intensity"], thres=thres)
    peak_frequencies = dataframe.iloc[center_indexes]["Frequency"]
    # Calculate the center frequency as the average
    center = np.average(peak_frequencies)
    if verbose is True:
        print("Center frequency at " + str(center))
    dataframe["Offset Frequency"] = dataframe["Frequency"] - center

def plot_chirp(chirpdf, catfiles=None):
    """ Function to perform interactive analysis with a chirp spectrum, as well
        as any reference .cat files you may want to provide.
        This is not designed to replace SPECData analysis, but simply to perform
        some interactive viewing of the data.

        The argument `catfiles` is supplied as a dictionary; where the keys are
        the names of the species, and the items are the paths to the .cat files
    """

    # Generate the experimental plot first
    plots = list()
    exp_trace = go.Scatter(
        x = chirpdf["Frequency"],
        y = chirpdf["Intensity"],
        name = "Experiment"
    )

    plots.append(exp_trace)
    if catfiles is not None:
        # Generate the color palette, and remove the alpha value from RGBA
        color_palette = plt.cm.spectral(np.linspace(0., 1., len(catfiles)))[:,:-1]
        # Loop over each of the cat files
        for color, species in zip(color_palette, catfiles):
            species_df = pc.pick_pickett(catfiles[species])
            plots.append(
                go.Bar(
                    x = species_df["Frequency"],
                    y = species_df["Intensity"] / species_df["Intensity"].min(),
                    name = species,
                    marker = {
                        # Convert the matplotlib rgb color to hex code
                        "color": colors.rgb2hex(color)
                    }
                    width = 1.,
                    opacity = 0.6,
                    yaxis = "y2"
                )
            )
    layout = go.Layout(
        yaxis={"title": ""},
        yaxis2={"title": "", "side": "right", "overlaying": "y", "range": [0., 1.]}
    )
    fig = go.Figure(data=plots, layout=layout)
    plot(fig)

def configure_colors(dataframe):
    """ Generates color palettes for plotting arbitrary number of SPECData
        assignments.
    """
    num_unique = len(dataframe["Assignment"].unique())
    return plt.cm.spectral(np.linspace(0., 1., num_unique))

def plot_specdata_mpl(dataframe):
    """ Function to display SPECData output using matplotlib.
        The caveat here is that the plot is not interactive, although it does
        provide an overview of what the assignments are. This is probably the
        preferred way if preparing a plot for a paper.
    """
    colors = configure_colors(dataframe)
    fig, exp_ax = plt.subplots(figsize=(10,6))

    exp_ax.vlines(dataframe["Exp. Frequency"], ymin=0., ymax=dataframe["Exp. Intensity"], label="Observations")

    exp_ax.set_yticks([])
    exp_ax.set_xlabel("Frequency (MHz)")

    assign_ax = exp_ax.twinx()
    current_limits = assign_ax.get_xlim()
    for color, assignment in zip(colors, dataframe["Assignment"].unique()):
        trunc_dataframe = dataframe[dataframe["Assignment"] == assignment]
        assign_ax.vlines(
            trunc_dataframe["Frequency"],
            ymin=np.negative((trunc_dataframe["Intensity"] / trunc_dataframe["Intensity"].max())),
            ymax=0.,
            alpha=0.5,
            label=assignment,
            color=color
        )
    exp_ax.hlines(0., 0., 30000.,)
    assign_ax.set_ylim([1., -1.])
    exp_ax.set_ylim([1., -1.])
    assign_ax.set_xlim(current_limits)

    assign_ax.set_yticks([])
    assign_ax.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=True)

def plot_specdata_plotly(dataframe):
    """ Interactive SPECData result plotting using plotly.
        The function will automatically swap between spectra and peaks by
        inspecting the number of data points we have.
    """
    if len(dataframe) >= 10000:
        exp_plot_function = go.Scatter
    else:
        exp_plot_function = go.Bar
    plots = list()
    plots.append(
        # Plot the experimental data
        exp_plot_function(
            x = dataframe["Exp. Frequency"],
            y = dataframe["Exp. Intensity"],
            name = "Experiment",
            width = 1.,
            opacity = 0.6
        )
    )
    # Use Matplotlib function to generate a colourmap
    color_palette = configure_colors(dataframe)
    # Loop over the colours and assignments
    for color, assignment in zip(color_palette, dataframe["Assignment"].unique()):
        # Truncate the dataframe to only hold the assignment
        trunc_dataframe = dataframe[dataframe["Assignment"] == assignment]
        plots.append(
            go.Bar(
                x=trunc_dataframe["Frequency"],
                y=np.negative((trunc_dataframe["Intensity"] / trunc_dataframe["Intensity"].max())),
                name=assignment,
                width=1.,
                marker={
                    # Convert the matplotlib color array to hex code
                    "color": colors.rgb2hex(color[:-1])
                }
            )
        )
    layout = go.Layout(
        yaxis={"title": "Intensity"},
        xaxis={"title": "Frequency (MHz)"}
    )
    fig = go.Figure(data=plots, layout=layout)
    plot(fig)
