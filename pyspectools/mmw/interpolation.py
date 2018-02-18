
import numpy as np
import pandas as pd

"""
    Routines for performing Shepard-interpolation with
    simple inverse-distance weighting.
    
    The basic usage of these routines is to feed in a list
    of Pandas dataframes, which consists of your experimental
    data. 
"""



def idw(x, xn, p=6, threshold=1.):
    """
        Inverse-distance weighting function
        For a given value of x with known data, and the interpolating
        x (xn), return the weighting value.
    """
    if x == xn:
        weight = 1e60
    else:
        weight = 1. / np.abs(x - xn)**p
        if weight < threshold:
            # Return a zero if the value is sufficiently small
            weight = 0.
    return weight
    

def vidw(x, xn, p=6, threshold=1.):
    """
        Vectorized form of the inverse-distance weighting function
    """
    func = np.vectorize(idw)
    return func(x, xn, p, threshold)


def find_nearest(dataframes, x, npoints=5):
    """
        Function that will take a list of dataframes, and
        return slices of the dataframes containing a specified
        number of closest points.
    """
    nearest = [
        df.loc[(df["Frequency"] - x).abs().argsort()[:npoints]] for df in dataframes
    ]
    return nearest


def find_nearest_nieu(dataframes, x, xrange=3., npoints=5):
    """
        New function that will return only data that is within a
        specified range, and within that the closest values.
    """
    nearest = list()
    
    for df in dataframes:
        # Find data within range
        temp_df = df.loc[(df["Frequency"] > x - xrange) & (df["Frequency"] < x + xrange)]
        # Only append if there is data inside the dataframe
        if temp_df.empty is False:
            nearest.append(temp_df.loc[(df["Frequency"] - x).abs().argsort()[:npoints]])
    return nearest


def shep_interp(dataframes, xn, col="Intensity", p=6, threshold=1.):
    """
        Calculate the Shepard interpolation of a point
    """
    # Loop over all selected data and calculate their weights
    weights = list()
    for df in dataframes:
        df["weight"] = vidw(
            df["Frequency"],
            xn,
            p,
            threshold
        )
        # Take only values that are not zero
        #df = df.loc[df["weight"] != 0.]
        weights.extend([value for value in df["weight"].values if value != 0.])
    weights = np.array(weights)
    cumint = 0.
    # Loop over data again, weighting the intensity and
    # calculating the interpolated intensity
    for df in dataframes:
        df["weighted-Intensity"] = df["weight"] * df[col]
        cumint+=df["weighted-Intensity"].sum()
    return cumint / np.sum(weights)

        
def calc_shep_interp(dataframes, xnew, col="Intensity", xrange=10., p=4., threshold=1., npoints=15):
    """
        Main function for calculating the Shepard interpolation for
        a given set of existing data, and an array corresponding
        to the new set of x data.
        
        dataframes - a list comprising pandas DataFrames, containing x/y data with
        column names ["Frequency", "Intensity"]
        xnew - the new x values to interpolate into. Use np.arange/np.linspace to
        generate.
        xrange - the frequency radius (in units of whatever frequency units are in data)
        to search for neighbours. Also defines the rate at which the neighbourlist is updated.
        p - the decay rate of the inverse distance weighting; should be even values.
        threshold - minimum weight value to be considered in the interpolation
        npoints - the maximum number of data points to use in the interpolation; smaller
        numbers mean crappy interpolation, larger means expensive.
        
        Returns the new y values
    """
    ynew = np.zeros(xnew.size)
    nele = ynew.size
    for index, x in enumerate(xnew):
        # Every 100 MHz, we will print a notification to tell you we're doing something...
        if index / nele * 100. % 10. == 0.:
            print(str(index / nele * 100.) + "% complete.")
        # Find nearest neighbours. This step will only be done every few steps as it
        # is the most computationally expensive part of the algorithm
        # The update frequency depends on the radius we look for neighbours, specified
        # with xrange.
        if index % xrange == 0.:
            neighbour_list = find_nearest_nieu(
                dataframes,
                x,
                xrange,
                npoints
            )
        # Calculate the y value based on nearest neighbours
        ynew[index]+=shep_interp(
            neighbour_list,
            x,
            col,
            p,
            threshold
        )
    return ynew