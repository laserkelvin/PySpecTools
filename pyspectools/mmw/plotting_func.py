from plotly.offline import iplot, plot, init_notebook_mode
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import colorlover as cl
import numpy as np

plt.style.use("seaborn")


def static_comparison(frequencies, sample_on_df, sample_off_df, window_size=0.0001):
    """
        Method for plotting the sample on-off spectra side-by-side.
        Takes a list of frequencies, which will be 
    """
    fig, axarray = plt.subplots(len(frequencies), 2, figsize=(12,24))
    
    for row_index, frequency in enumerate(frequencies):
        xrange = [
            frequency * (1. - window_size),
            frequency * (1. + window_size)
        ]
        for col_index, df in enumerate([sample_on_df, sample_off_df]):
            slice_df = df.loc[(df["Frequency"] > xrange[0]) & (df["Frequency"] < xrange[1])]
            slice_df["Frequency"] -= frequency
            axarray[row_index, col_index].plot(
                slice_df["Frequency"],
                slice_df["Field off"],
                label="Field Off",
                alpha=0.4
            )
            axarray[row_index, col_index].plot(
                slice_df["Frequency"],
                slice_df["Field on"],
                label="Field On",
                alpha=0.4
            )
            axarray[row_index, col_index].plot(
                slice_df["Frequency"],
                slice_df["Off - On"],
                label="Off - On",
                alpha=1.
            )
            axarray[row_index, col_index].legend()
            axarray[row_index, col_index].set_title(np.round(frequency, decimals=4))
            if row_index == len(frequencies) - 1:
                axarray[row_index, col_index].set_xlabel("Frequency (MHz)")
                
    fig.tight_layout()
    return fig, axarray


def plot_spectrum(dataframe):
    """ Function used to plot a dataframe using Plotly.
        
        Uses colorlover to generate palettes.

        Args: dataframe - pandas dataframe containing a frequency
        column. Every other column is treated as an intensity column
    """
    plots = list()
    keys = [key for key in dataframe.keys() if key != "Frequency"]
    if len(keys) < 3:
        # If there are fewer than 3 plots, colorlover doesn't have
        # a coded case and so the colors are done manually
        color_palette = ["#e41a1c", "#377eb8"]
    else:
        # Use color lover palettes
        color_palette = cl.to_rgb(cl.scales[str(len(keys))]["qual"]["Set1"])
    for key, color in zip(keys, color_palette):
        # Loop over all the dataframe columns and colors
        plots.append(
            go.Scatter(
                x=dataframe["Frequency"],
                y=dataframe[key],
                name=key,
                marker = {
                   "color": color
               },
            )
        )
            
    layout = go.Layout(
        xaxis={"title": "Frequency (MHz)", "tickformat": "0.2f"},
        yaxis={"title": ""},
        autosize=False,
        height=800,
        width=1000,
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
    )
    
    fig = go.Figure(data=plots, layout=layout)
    
    iplot(fig)
    return fig


def id_survey(frequency, intensity, catdict):
    """
        Function for plotting predicted lines as stick spectra
        on top of a broadband survey spectrum.
        
        Frequency and intensity are 1D arrays of the spectrum
        frequency and intensity.
        
        catdict is a dictionary with keys corresponding to
        the name of the molecule, and values corresponding
        to dataframes of the parsed .cat file.
    """
    plots = list()
    
    # Plot out the experiment first
    
    plots.append(
        go.Scatter(
            x=frequency,
            y=intensity,
            marker={"color": (0., 0., 0.)}
        )
    )
    
    # Set the colour palette, and interpolate colours to
    # accommodate for the number of cat files
    spectral_palette = cl.scales["8"]["div"]["Spectral"]
    colors = cl.interp(spectral_palette, len(catdict))
    
    for name, color in zip(catdict, colors):
        plots.append(
            go.Bar(
                x=catdict[name]["Frequency"],
                y=np.abs(catdict[name]["Intensity"]),
                name=name,
                width=2.,
                marker={"color": color}
            )
        )
    
    layout = go.Layout(
        xaxis={
            "title": "Frequency (MHz)",
            "range": [np.min(frequency), np.max(frequency)],
            "tickformat": "0.2f"
        },
        yaxis={"title": ""},
        autosize=False,
        height=800,
        width=1000,
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
    )
    
    fig = go.Figure(data=plots, layout=layout)
    
    iplot(fig)
    return fig


def save_plot(fig, filename):
    plot(
        fig,
        filename=filename,
        show_link=False,
        auto_open=False
    )
