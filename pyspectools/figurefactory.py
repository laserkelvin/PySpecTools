import os
from warnings import warn

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib import colors as cl
from plotly import graph_objs as go
from plotly.offline import plot
from plotly import tools
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.io import curdoc
from bokeh.themes import Theme

from pyspectools import routines


"""
    Commonly used formatting options

    These are generally parts of Matplotlib that I commonly change,
    but always have to look up stack overflow to find out how to do...
"""


def strip_spines(spines, axis):
    # Function for removing the spines from an axis.
    for spine in spines:
        axis.spines[spine].set_visible(False)


def no_scientific(axis):
    # Turns off scientific notation for the axes
    axis.get_xaxis().get_major_formatter().set_useOffset(False)
    axis.get_yaxis().get_major_formatter().set_useOffset(False)


def format_ticklabels(axis):
    # Adds commas to denote thousands - quite useful for
    # publications
    axis.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )


"""
    Specific figure types

    These include recipes/routines to generate commonly used
    figures, providing the data is provided in a digestable
    way for the scripts.

    These include:
        - Polyad diagrams
        - Generic energy diagrams
        - Adding images to a matplotlib figure
"""


def make_pes(x, energies, width=5):
    """ Function to create a smooth PES, where stationary 
        points are connected.
        
        Basically makes a linearly spaced x array which
        pads x-axis density to create the horizontal line
        for the stationary points
        
        Optional arguments are how many x-points to 
        pad with on either side.
    """
    new_x = np.array([])
    new_energies = list()
    for xvalue, energy in zip(x, energies):
        new_x = np.append(
            new_x,
            np.linspace(xvalue - (width * 0.05), xvalue + (width * 0.05), width * 2),
        )
        new_energies.append([energy] * width * 2)
    return new_x, np.array(new_energies).flatten()


def calc_vibE(quant_nums, vibrations):
    """
        Function that will calculate a 1D array of
        vibrational energies, based on a nested list
        of quantum numbers and a list of vibrational
        frequencies.
    """
    energies = list()
    for state in quant_nums:
        energies.append({"state": state, "energy": np.sum(state * vibrations)})
    return energies


def generate_x_coord(quant_nums, energies):
    """
        Function to generate the x coordinate for plotting
        based on the quantum numbers specified.
    """
    # Get boolean mask, and find the unique combinations
    unique_combos = np.unique(quant_nums > 0, axis=0)
    n_unique = len(unique_combos)
    cat_dict = {x: list() for x in np.arange(n_unique)}
    for item in energies:
        index = np.where((unique_combos == (item["state"] > 0)).all(axis=1))[0][0]
        cat_dict[index].append(item)
    return cat_dict


def make_elevel_plot(cat_dict, axis, color, maxE, alpha=1.0):
    """
        make_elevel_plot is used to create energy level diagrams
        with Matplotlib axis objects. This is considered slightly
        lower level: you should call the higher level wrapper functions
        like vib_energy_diagram instead!

        The function will loop over every transition that has
        been categorized/binned into an x-axis index, and sub-
        sequently plot each x-bin individually.

        The input arguments are:

        cat_dict: dict-like with keys as the x-axis values, and
        the items are dictionaries containing the energy and
        quantum number configuration.
    """
    # Spacing defines the x-axis unit spacing
    spacing = 5
    # Defines the width of the levels
    width = 1.0
    # Loop over every state
    for index in cat_dict:
        xs = list()
        ys = list()
        annotations = list()
        for item in cat_dict[index]:
            xs.append(index)
            ys.append(item["energy"])
            annotations.append(str(tuple(item["state"])).replace(",", ""))
        axis.hlines(
            ys,
            [x - (width / 2.0) + spacing for x in xs],
            [x + (width / 2.0) + spacing for x in xs],
            color=color,
            alpha=alpha,
        )
        # Loop over configurations and annotate the energy levels
        # with their quantum numbers
        for x, y, text in zip(xs, ys, annotations):
            # Providing the value is less than the maximum specified
            # energy, add the annotation
            if y < maxE:
                axis.text(
                    x + spacing,
                    y + 30.0,
                    text,
                    horizontalalignment="center",
                    color=color,
                    alpha=alpha,
                    size=8.0,
                )


def add_image(axis, filepath, zoom=0.15, position=[0.0, 0.0]):
    """
        Function to add an image annotation to a specified axis.

        Takes the matplotlib axis and filepath to the image as input,
        and optional arguments for the scaling (zoom) and position of
        the image in axis units.
    """
    image = OffsetImage(plt.imread(filepath, format="png"), zoom=zoom)
    image.image.axes = axis

    box = AnnotationBbox(
        image, position, xybox=position, xycoords="data", frameon=False
    )
    axis.add_artist(box)


def vib_energy_diagram(
    quant_nums, vibrations, maxV=2, maxE=3000.0, useFull=True, image=None, imagesize=0.1
):
    """
        Function that will generate a vibrational energy diagram.

        This function wraps the make_elevel_plot function!

        Input arguments are

        quant_nums: A 2D numpy array of quantum numbers, where each row is
        a single configuration.
        vibrations: A list of vibrational frequencies.
        maxV: The maximum quanta for considering predicted frequencies.
        maxE: Maximum energy for plotting.
        useFull: Boolean for flagging whether predict frequencies are plotted.
        image: String-like for specifying path to an image.
    """
    # Generate a list of possible quanta from 0 to maxV
    full_quant = np.arange(0, maxV)
    # Create a generator that will provide every possible configuration
    # of quantum numbers
    full_combo = np.array(list(product(full_quant, repeat=len(vibrations))))
    # Calculate the energies of all possible configurations
    full_energies = calc_vibE(full_combo, vibrations)
    full_cat_dict = generate_x_coord(full_combo, full_energies)
    energies = calc_vibE(quant_nums, vibrations)
    # useFull denotes whether or not to use the predicted combination
    # energies. If False, we only display the vibrations observed
    if useFull is True:
        cat_dict = generate_x_coord(full_combo, energies)
    else:
        cat_dict = generate_x_coord(quant_nums, energies)

    # Initialize the figure object
    fig, ax = plt.subplots(figsize=(5, 5.5))
    # If we want to show predictions, plot them up too
    if useFull is True:
        make_plot(full_cat_dict, ax, "black", maxE, 0.6)
    # Call function to plot up diagram
    make_plot(cat_dict, ax, "#e41a1c", maxE)

    # Set various labelling
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.set_xlabel("Vibrational state")
    ax.set_ylabel("Energy (cm$^{-1}$)")
    ax.set_ylim([-50.0, maxE])

    minx = ax.get_xlim()[0]
    maxx = ax.get_xlim()[1]

    # Annotate the fundamentals
    for vibration in vibrations:
        ax.hlines(
            vibration,
            minx,
            maxx,
            color="#377eb8",
            linestyle="--",
            zorder=0.0,
            alpha=0.3,
        )
        # If an image filepath is specified, then plot it up on the edge
        if image is not None:
            add_image(ax, image, zoom=imagesize, position=[maxx - 0.5, vibration])
    fig.tight_layout()

    return fig, ax


def overlay_dr_spectrum(
    dataframe, progressions, freq_col="Frequency", int_col="Intensity", **kwargs
):
    layout = define_layout("Frequency (MHz)", "Intensity")
    fig = go.FigureWidget(layout=layout)

    fig.add_scattergl(
        x=dataframe[freq_col], y=dataframe[int_col], name="Observation", opacity=0.4
    )

    colors = generate_colors(len(progressions), cmap=plt.cm.tab10)
    level = 2.0

    for index, (progression, color) in enumerate(zip(progressions, colors)):
        mask = np.where(progression <= np.max(dataframe[freq_col]))
        progression = progression[mask]
        indices = np.array(
            [routines.find_nearest(dataframe[freq_col], freq) for freq in progression]
        )
        indices = indices[:, 1]
        y = dataframe[int_col].iloc[indices] * 1.2
        fig.add_scattergl(
            x=progression,
            y=y,
            marker={"color": color},
            mode="markers+lines",
            hoverinfo="name+x",
            name="Progression {}".format(index),
        )
    return fig


def dr_network_diagram(connections, **kwargs):
    """
    Use NetworkX to create a graph with nodes corresponding to cavity
    frequencies, and vertices as DR connections.
    The color map can be specified by passing kwargs.
    :param connections: list of 2-tuples corresponding to pairs of connections
    :return
    """
    graph = nx.Graph()
    nodes = [graph.add_node(frequency) for frequency in np.unique(connections)]
    vertices = [graph.add_edge(*pair) for pair in connections]
    # Generate positions based on the shell layout that's typical of DR connections
    # Frequencies are sorted in anti-clockwise order, starting at 3 o'clock
    positions = nx.shell_layout(graph)

    color_kwarg = {"cmap": plt.cm.tab10}
    if "cmap" in kwargs:
        color_kwarg.update(**kwargs)

    coords = np.array(list(positions.values()))
    connected = list(nx.connected_components(graph))
    colors = generate_colors(len(connected), **color_kwarg)

    fig_layout = {
        "height": 700.0,
        "width": 700.0,
        "autosize": True,
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "ticks": "",
            "showticklabels": False,
        },
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "ticks": "",
            "showticklabels": False,
        },
        "showlegend": False,
    }
    fig = go.FigureWidget(layout=fig_layout)
    # Draw the nodes
    fig.add_scattergl(
        x=coords[:, 0],
        y=coords[:, 1],
        text=list(np.unique(connections)),
        hoverinfo="text",
        mode="markers",
    )
    # Draw the vertices
    for connectivity, color in zip(connected, colors):
        # Get all of the coordinates associated with edges within a series
        # of connections
        coords = np.array([positions[node] for node in sorted(connectivity)])
        fig.add_scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="lines",
            hoverinfo=None,
            name="",
            opacity=0.4,
            marker={"color": color},
        )
    return fig, connected


def init_plotly_subplot(nrows, ncols, **kwargs):
    """
    Initialize a Plotly subplot.
    :param nrows: number of rows for the subplot
    :param ncols: number of columns for the subplot
    :return: plotly FigureWidget object
    """
    subplot = tools.make_subplots(rows=nrows, cols=ncols, **kwargs)
    fig = go.FigureWidget(subplot)
    return fig


def stacked_plot(
    dataframe, frequencies, freq_range=0.002, freq_col="Frequency", int_col="Intensity"
):
    """
    Create a Loomis-Wood style plot via a list of frequencies, and a broadband
    spectrum. The keyword freq_range will use a percentage of the center
    frequency to extend the frequency range that gets plotted.
    :param dataframe: pandas DataFrame
    :param frequencies: iterable with float frequencies to use as centers
    :param freq_range: decimal percentage to specify the range to plot
    :param freq_col: str name for the column to use as the frequency axis
    :param int_col: str name for the column to use as the intensity axis
    :return fig: Plotly FigureWidget with the subplots
    """
    plot_func = go.Scattergl

    # Want the frequencies in ascending order, going upwards in the plot
    indices = np.where(
        np.logical_and(
            dataframe[freq_col].min() <= frequencies,
            frequencies <= dataframe[freq_col].max(),
        )
    )
    # Plot only frequencies within band
    frequencies = frequencies[indices]
    frequencies = np.sort(frequencies)[::-1]
    nplots = len(frequencies)

    titles = tuple("{:.0f} MHz".format(frequency) for frequency in frequencies)
    fig = init_plotly_subplot(
        nrows=nplots,
        ncols=1,
        **{"subplot_titles": titles, "vertical_spacing": 0.15, "shared_xaxes": True},
    )
    for index, frequency in enumerate(frequencies):
        # Calculate the offset frequency
        dataframe["Offset " + str(index)] = dataframe[freq_col] - frequency
        # Range as a fraction of the center frequency
        freq_cutoff = freq_range * frequency
        sliced_df = dataframe.loc[
            (dataframe["Offset " + str(index)] > -freq_cutoff)
            & (dataframe["Offset " + str(index)] < freq_cutoff)
        ]
        # Plot the data
        trace = plot_func(
            x=sliced_df["Offset " + str(index)], y=sliced_df[int_col], mode="lines"
        )
        # Plotly indexes from one because they're stupid
        fig.add_trace(trace, index + 1, 1)
        fig["layout"]["xaxis1"].update(
            range=[-freq_cutoff, freq_cutoff],
            title="Offset frequency (MHz)",
            showgrid=True,
        )
        fig["layout"]["yaxis" + str(index + 1)].update(showgrid=False)
    fig["layout"].update(autosize=True, height=1000, width=900, showlegend=False)
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
    exp_trace = go.Scattergl(
        x=chirpdf["Frequency"], y=chirpdf["Intensity"], name="Experiment"
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
                    width=1.0,
                    opacity=0.6,
                    yaxis="y2",
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
        yaxis2={"title": "", "side": "right", "overlaying": "y", "range": [0.0, 1.0]},
    )
    fig = go.FigureWidget(data=plots, layout=layout)

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
    traces.append(plot_column(spec_df, col, color=next(colors)))
    # Loop over the assignments
    for molecule in molecules:
        sliced_df = assignments_df.loc[assignments_df["Chemical Name"] == molecule]
        traces.append(plot_bar_assignments(sliced_df, next(colors)))
    layout = define_layout()
    layout["yaxis"] = {"title": "Experimental Intensity"}
    # Specify a second y axis for the catalog intensity
    layout["yaxis2"] = {
        "title": "CDMS/JPL Intensity",
        "overlaying": "y",
        "side": "right",
        "type": "log",
        "autorange": True,
    }
    figure = go.Figure(data=traces, layout=layout)
    plot(figure)
    return figure


def generate_colors(n, cmap=plt.cm.Spectral, hex=True):
    """
    Generate a linearly spaced color series using a colormap from
    Matplotlib. The colors can be returned as either RGB values
    or as hex-codes using the `hex` flag.

    Parameters
    ----------
    n : int
        Number of colours to generate
    cmap : str or `matplotlib.colors.LinearSegementedColomap`, optional
        Specified colormap to interpolate. If a str is provided, the function
        will try to look for it in the available matplotlib colormaps.
    hex : bool, optional
        If True, hex colors are returned. Otherwise, RGB values.
    Returns
    -------
    colors
        List of hex or RGB codes
    """
    # In the case that a string is passed, use the `get_cmap` function instead
    if type(cmap) == str:
        try:
            cmap = plt.cm.get_cmap(cmap)
        except ValueError:
            warn(f"{cmap} not found in Matplotlib, defaulting to Spectral.")
            pass
    colormap = cmap(np.linspace(0.0, 1.0, n))
    if hex is True:
        colors = [cl.rgb2hex(color) for color in colormap]
    else:
        colors = colormap
    return colors


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
        y=10 ** species_df["CDMS/JPL Intensity"],
        name=molecule,
        text=species_df["Resolved QNs"],
        marker={"color": color},
        width=0.25,
        yaxis="y2",
        opacity=0.9,
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
        x=dataframe["Frequency"], y=dataframe[col], name=name, marker={"color": color}
    )
    # If a layout is supplied, plot the figure
    if layout:
        figure = go.Figure(data=[trace], layout=layout)
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
        xaxis={"title": xlabel, "tickformat": ".,"},
        yaxis={"title": ylabel},
        autosize=True,
        height=650.0,
        width=850.0,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Roboto", size=14, color="#000000"),
        annotations=list(),
    )
    return layout


def save_plot(fig, filename, js=True):
    """
        Method for exporting a plotly figure with interactivity.
        This method does inject the plotly.js code by default, and so will
        result in relatively large files. Use `save_html` instead.
    """
    plot(fig, filename=filename, show_link=False, auto_open=False, include_plotlyjs=js)


def cfa_cmap(nsteps=100):
    """
    Generate a Matplotlib colormap with the CfA branding colors. Performs a linear interpolation from the CfA red to
    the so-called CfA violet.

    Parameters
    ----------
    nsteps - int
        Number of steps to take in the interpolation; i.e. number of colors to return

    Returns
    -------
    LinearSegementedColormap instance
    """
    colors = [(141, 0, 52), (43, 53, 117)]
    cm = cl.LinearSegmentedColormap("cfa", colors, N=nsteps)
    return cm


def pandas_bokeh_table(dataframe, html=False, **kwargs):
    """
    Convert a Pandas DataFrame to a Bokeh DataTable object.
    Columns will be automatically generated based on the DataFrame keys.
    The `html` flag can be used to specify whether or not an HTML representation is returned.

    Additional kwargs are passed into the `file_html` function, and is only used when `html`
    is True.

    Parameters
    ----------
    dataframe : pandas dataframe
        Pandas DataFrame to convert into a DataTable object
    html : bool, optional
        If True, function will return a string of the HTML code for embedding
    kwargs
        Additional kwargs are passed into the HTML conversion

    Returns
    -------
    DataTable object if html is False, otherwise str
    """
    source = ColumnDataSource(dataframe)
    columns = [TableColumn(field=key, title=key.capitalize()) for key in dataframe]
    table = DataTable(source=source, columns=columns, **kwargs)
    if html is True:
        return file_html(table, CDN)
    else:
        return table


def init_bokeh_figure(yml_path=None, **kwargs):
    """
    Initialize a Bokeh Figure object. This code is fairly low level, and is
    used simply as a matter of convenience for supplying a default theme even
    without the user explicitly providing one.

    Kwargs are passed into the creation of the figure, and so further
    customization like title, axis limits, etc. should be supplied this way.

    Parameters
    ----------
    yml_path : str or None, optional
        If None, uses the PySpecTools default bokeh styling. If a str is
        provided then it should correspond to the filepath to a YAML file.
    kwargs
        Kwargs are passed into the creation of the Figure object

    Returns
    -------
    Bokeh Figure object
    """
    if yml_path is None:
        # Use the default stylesheet
        yml_path = os.path.expanduser("~") + "/.pyspectools/bokeh.yml"
    # Set the document theme
    doc = curdoc()
    doc.theme = Theme(filename=yml_path)
    default_params = {
        "tools": "crosshair,pan,wheel_zoom,box_zoom,reset,box_select," "lasso_select",
        "outline_line_color": "black",
        "output_backend": "webgl",
    }
    default_params.update(**kwargs)
    fig = figure(**default_params)
    return fig
