
import matplotlib as mpl
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np

mpl.style.use("seaborn")

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
                np.linspace(
                    xvalue - (width * 0.05), 
                    xvalue + (width * 0.05), 
                    width * 2)
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
        energies.append(
                {
                "state": state,
                "energy": np.sum(state * vibrations)
            }
        )
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
        cat_dict[index].append(
            item
        )
    return cat_dict


def make_elevel_plot(cat_dict, axis, color, maxE, alpha=1.):
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
    width = 1.
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
            [x - (width / 2.) + spacing for x in xs],
            [x + (width / 2.) + spacing for x in xs],
            color=color,
            alpha=alpha
        )
        # Loop over configurations and annotate the energy levels
        # with their quantum numbers
        for x, y, text in zip(xs, ys, annotations):
            # Providing the value is less than the maximum specified
            # energy, add the annotation
            if y < maxE:
                axis.text(
                    x + spacing,
                    y + 30.,
                    text,
                    horizontalalignment="center",
                    color=color,
                    alpha=alpha,
                    size=8.
                )


def add_image(axis, filepath, zoom=0.15, position=[0., 0.]):
    """
        Function to add an image annotation to a specified axis.

        Takes the matplotlib axis and filepath to the image as input,
        and optional arguments for the scaling (zoom) and position of
        the image in axis units.
    """
    image = OffsetImage(plt.imread(filepath, format="png"), zoom=zoom)
    image.image.axes = axis

    box = AnnotationBbox(image, position,
                        xybox=position,
                        xycoords='data',
                        frameon=False
                        )
    axis.add_artist(box)


def vib_energy_diagram(quant_nums, vibrations, maxV=2, maxE=3000.,
                       useFull=True, image=None, imagesize=0.1):
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
    fig, ax = plt.subplots(figsize=(5,5.5))
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
    ax.set_ylim([-50., maxE])

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
            zorder=0.,
            alpha=0.3
        )
        # If an image filepath is specified, then plot it up on the edge
        if image is not None:
            add_image(
                ax,
                image,
                zoom=imagesize,
                position=[
                    maxx - 0.5,
                    vibration
                ]
            )
    fig.tight_layout()

    return fig, ax
