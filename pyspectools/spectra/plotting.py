""" plotting.py

    Functions for plotting up chirp data that requires interactivity based
    on Plotly.

    The idea is to have some low level wrappers that will plot whatever data in
    pandas dataframes, then have some specific higher level functions that will
    plot specific stuff (e.g. artifact comparison, assignments).
"""

from plotly.offline import plot, init_notebook_mode, iplot
from plotly import tools
import plotly.graph_objs as go
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as cl


def stacked_plot(dataframe, frequencies, freq_range=0.002):
    # Function for generating an interactive stacked plot.
    # This form of plotting helps with searching for vibrational satellites.
    # Input is the full dataframe containing the frequency/intensity data,
    # and frequencies is a list containing the centering frequencies.
    # The frequency range is specified as a percentage of frequency, so it
    # will change the range depending on what the centered frequency is.
    nplots = len(frequencies)
    plot_func = go.Scatter

    # Want the frequencies in ascending order, going upwards in the plot
    frequencies = np.sort(frequencies)[::-1]

    titles = tuple("{:.4f} MHz".format(frequency) for frequency in frequencies)
    fig = tools.make_subplots(
        rows=nplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=titles,
    )

    for index, frequency in enumerate(frequencies):
        # Calculate the offset frequency
        dataframe["Offset " + str(index)] = dataframe["Frequency"] - frequency
        # Range as a fraction of the center frequency
        freq_cutoff = freq_range * frequency
        sliced_df = dataframe.loc[
            (dataframe["Offset " + str(index)] > -freq_cutoff) & (dataframe["Offset " + str(index)] < freq_cutoff)
        ]
        # Plot the data
        trace = plot_func(
            x=sliced_df["Offset " + str(index)],
            y=sliced_df["Intensity"],
            text=sliced_df["Frequency"],
            mode="lines"
        )
        # Plotly indexes from one because they're stupid
        fig.append_trace(trace, index + 1, 1)
        fig["layout"]["xaxis1"].update(
            range=[-freq_cutoff, freq_cutoff],
            title="Offset frequency (MHz)",
            showgrid=True
        )
        fig["layout"]["yaxis" + str(index + 1)].update(showgrid=False)
    fig["layout"].update(
        autosize=False,
        height=800,
        width=1000,
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
        showlegend=False
    )
    iplot(fig)
    return fig


def plot_catchirp(chirpdf, catfiles=None):
    """ Function to perform interactive analysis with a chirp spectrum, as well
        as any reference .cat files you may want to provide.
        This is not designed to replace SPECData analysis, but simply to
        perform some interactive viewing of the data.

        The argument `catfiles` is supplied as a dictionary; where the keys are
        the names of the species, and the items are the paths to the .cat files
    """

    # Generate the experimental plot first
    plots = list()
    exp_trace = go.Scatter(
        x=chirpdf["Frequency"],
        y=chirpdf["Intensity"],
        name="Experiment"
    )

    plots.append(exp_trace)
    if catfiles is not None:
        # Generate the color palette, and remove the alpha value from RGBA
        color_palette = generate_colors(len(catfiles))
        # Loop over each of the cat files
        for color, species in zip(color_palette, catfiles):
            species_df = pc.pick_pickett(catfiles[species])
            plots.append(
                go.Bar(
                    x=species_df["Frequency"],
                    y=species_df["Intensity"] / species_df["Intensity"].min(),
                    name=species,
                    marker={
                        # Convert the matplotlib rgb color to hex code
                        "color": color
                    },
                    width=1.,
                    opacity=0.6,
                    yaxis="y2"
                )
            )
    layout = go.Layout(
        autosize=False,
        height=600,
        width=900,
        xaxis={"title": "Frequency (MHz)"},
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
        yaxis={"title": ""},
        yaxis2={
            "title": "",
            "side": "right",
            "overlaying": "y",
            "range": [0., 1.]
        }
    )
    fig = go.Figure(data=plots, layout=layout)
    iplot(fig)

    return fig


def plot_df(dataframe, cols=None, **kwargs):
    """ Function that wraps around the lower level function plot_column.
        Will plot every column in a dataframe against the Frequency, unless
        specific column names are provided.

        Input arguments:
        dataframe - pandas dataframe object, with every column as intensity
        except "Frequency"
        cols - NoneType or tuple-like: if None, every column is plotted.
        Alternatively, an iterable is provided to specify which columns are
        plotted.
        Optional arguments are passed into define_layout, which will define
        the axis labels, or into the color map generation
    """
    if cols is None:
        cols = [key for key in dataframe.keys() if key != "Frequency"]
    if len(cols) < 4:
        colors = ["#66c2a5", "#fc8d62"]
    else:
        colors = generate_colors(len(cols), **kwargs)
    # Generate the plotly traces
    traces = [
        plot_column(dataframe, col, color=color) for col, color in zip(cols, colors)
    ]
    layout = define_layout(**kwargs)
    # Generate figure object
    figure = go.Figure(data=traces, layout=layout)
    iplot(figure)
    return figure


def plot_assignment(spec_df, assignments_df, col="Intensity"):
    """ Function for plotting spectra with assignments. The assumption is that
        the assignments routines are ran prior too this, and so the function
        simply takes a dataframe of chosen molecules and plots them alongside
        the experimental spectrum, color coded by the molecule

        Input argument:
        spec_df - dataframe holding the experimental data
        assignments_df - dataframe produced from running assignments
    """
    # Get a list of unique molecules
    molecules = assignments_df["Chemical Name"].unique()
    # The ttal number of traces are the number of unique molecules, the traces
    # in the experimental data minus the frequency column
    nitems = len(molecules) + 1
    colors = color_iterator(nitems)
    traces = list()
    # Loop over the experimental data
    traces.append(
        plot_column(
            spec_df,
            col,
            color=next(colors)
        )
    )
    # Loop over the assignments
    for molecule in molecules:
        sliced_df = assignments_df.loc[assignments_df["Chemical Name"] == molecule]
        traces.append(
            plot_bar_assignments(
                sliced_df,
                next(colors)
            )
        )
    layout = define_layout()
    layout["yaxis"] = {
        "title": "Experimental Intensity"
    }
    # Specify a second y axis for the catalog intensity
    layout["yaxis2"] = {
        "title": "CDMS/JPL Intensity",
        "overlaying": "y",
        "side": "right",
        "type": "log",
        "autorange": True
    }
    figure = go.Figure(data=traces, layout=layout)
    iplot(figure)
    return figure


def generate_colors(n, cmap=plt.cm.Spectral):
    """ Simple function for generating a colour map """
    colormap = cmap(np.linspace(0., 1., n))
    colors = [cl.rgb2hex(color) for color in colormap]
    return colors#[:, :-1]


def color_iterator(n, **kwargs):
    """ Simple generator that will yield a different color each time.
        This is primarily designed when multiple plot types are expected.

        Input arguements:
        n - number of plots
        Optional kwargs are passed into generate_colors
    """
    index = 0
    colors = generate_colors(n, **kwargs)
    while index < n:
        yield colors[index]
        index += 1


def plot_bar_assignments(species_df, color="#fc8d62"):
    """ Function to generate a bar plot trace for a chemical species.
        These plots will be placed in the second y axis!

        Input arguments:
        species_df - a slice of an assignments dataframe, containing only
        one unique species
        Optional argument color is a hex code color; if nothing is given
        it just defaults to whatever
    """
    # We just want one of the molecules, not their life's story
    molecule = species_df["Chemical Name"].unique()[0]
    trace = go.Bar(
        x=species_df["Combined"],
        y=10**species_df["CDMS/JPL Intensity"],
        name=molecule,
        text=species_df["Resolved QNs"],
        marker={
            "color": color
            },
        width=0.25,
        yaxis="y2",
        opacity=0.9
    )
    return trace


def plot_column(dataframe, col, name=None, color=None, layout=None):
    """ A low level function for plotting a specific column of
        data in a pandas dataframe. This will assume that there
        is a column named "Frequency" in the dataframe.

        If a layout is not supplied, then the function will
        return a Plotly scatter object to be combined with other
        data. If a layout is given, then the data will be plot
        up directly.

        Input arguments:
        dataframe - pandas dataframe object
        col - str specifying the column used to plot
        layout - optional argument; if specified a plotly plot will be
        produced.
    """
    # If no legend name is provided, use the column
    if name is None:
        name = col
    # Generate the scatter plot
    if color is None:
        color = "#1c9099"
    trace = go.Scatter(
        x=dataframe["Frequency"],
        y=dataframe[col],
        name=name,
        marker={
            "color": color
        }
    )
    # If a layout is supplied, plot the figure
    if layout:
        figure = go.Figure(
            data=[trace],
            layout=layout
        )
        iplot(figure)
    else:
        return trace


def define_layout(xlabel="", ylabel=""):
    """ Function for generating a layout for plotly.
        Some degree of customization is provided, but generally sticking
        with not having to fuss around with plots.

        Input arguments:
        x/ylabel - str for what the x and y labels are to be
    """
    layout = go.Layout(
        xaxis={"title": xlabel, "tickformat": "0.1f"},
        yaxis={"title": ylabel, "tickformat": None},
        autosize=False,
        height=600.,
        width=1200.,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family='Roboto', size=18, color='#000000'),
        annotations=list()
    )
    return layout


def save_plot(fig, filename, js=True):
    """
        Method for exporting a plotly figure with interactivity.
        This method does inject the plotly.js code by default, and so will
        result in relatively large files. Use `save_html` instead.
    """
    plot(
        fig,
        filename=filename,
        show_link=False,
        auto_open=False,
        include_plotlyjs=js
    )
