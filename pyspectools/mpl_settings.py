
import matplotlib as mpl
from cycler import cycler
from matplotlib import pyplot as plt

mpl.style.use("seaborn")

params = {
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "legend.fontsize": "x-large",
    "figure.figsize": (12, 5.5),
    "axes.prop_cycle": cycler("color", ["#d53e4f", "#fc8d59", "#fee08b", "#e6f598", "#99d594", "#3288bd"])
}
mpl.rcParams.update(params)

def mpl_annotation(x, y, axis, text="Text"):
    return axis.text(x, y, text, size="x-large")


def mpl_scatter(x, y, yerr=None, facecolors="none", edgecolors="#feb24c", alpha=0.9, label=None):
    scatter_object = plt.scatter(
    x = x,
    y = y,
    facecolors = facecolors,
    edgecolors = edgecolors,
    alpha = alpha,
    label = label
    )
    return scatter_object


def mpl_plot(x, y, yerr=None, width=1., color="feb24c", alpha=0.9, label=None):
    plot_object = plt.plot(
    x = x,
    y = y,
    yerr = yerr,
    width = width,
    color = color,
    alpha = alpha,
    label = label
    )
    return plot_object


def strip_spines(spines, axis):
    # Function for removing the spines from an axis.
    for spine in spines:
        axis.spines[spine].set_visible(False)


def no_scientific(axis):
    # Turns off scientific notation for the axes
    axis.get_xaxis().get_major_formatter().set_useOffset(False)
    axis.get_yaxis().get_major_formatter().set_useOffset(False)
