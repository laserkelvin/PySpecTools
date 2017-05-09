
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import signal as spsig
import sys
import os

#
mpl.style.use("seaborn")

params = params = {
    "backend": "qt5agg",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "legend.fontsize": "x-large",
    "figure.figsize": (12, 5.5),
}
mpl.rcParams.update(params)

def find_peaks(xarray, yarray, snr=3):
    """
    Kyle's peak finding algorithm

    Return lists of peaks in yarray

    This function locates peaks in y array, returning lists containing
    their x values, y values, and signal-to-noise ratios.

    Peak finding is based on a noise estimator and 2nd derivative
    procedure. First, a noise model is constructed by breaking the
    y array into chunks. Within each chunk, the median noise level is
    iteratively located by calculating the median, removing all points
    10x greater than the median, and repeating until no more points are
    removed. After removal of these points, the mean and standard
    deviation are used to model the baseline offset and noise for that
    segment.

    A second derivative of the y array is calculated using a 6th order,
    11 point Savitzky Golay filtered version of yarray. This smoothed
    second derivative is used later in the peak finding routine.

    To locate peaks, the program loops over y array, calculating the SNR
    at each point. If the SNR is above the threshold, then the second
    derivative is scanned to see if the point is a local minimum by
    comparing its value to two 4-point windows ([i-2, i-1, i, i+1] and
    [i-1,i,i+1,i+2]). If either test returns true, then the point
    is flagged as a peak and added to the output arrays.

    Arguments:

        xarray -- Array-like x values

        yarray -- Array-like y values

        snr -- Minimum SNR for peak detection (default: 3)
    Returns outx, outy, outidx, outsnr, noise, baseline:

        outx -- Array of peak x values

        outy -- Array of peak y values

        outidx -- Array of peak indices

        outsnr -- Array of estimated signal-to-noise ratios

        noise -- Array containing 1 sigma noise estimate

        baseline -- Array containing baseline estimate


    """

    #compute smoothed 2nd derivative for strong peaks
    #if not self.quiet:
    #    print("Computing smoothed second derivative...")
    d2y = spsig.savgol_filter(yarray,11,6,deriv=2)

    #if not self.quiet:
    #    print("Building noise model...")

    #build noise model
    chunks = 10
    chunk_size = len(yarray)//chunks
    avg = []
    noise = []
    dat = []
    outnoise = np.empty(len(yarray))
    outbaseline = np.empty(len(yarray))
    for i in range(chunks):
        if i + 1 == chunks:
            dat = yarray[i*chunk_size:]
        else:
            dat = yarray[i*chunk_size:(i+1)*chunk_size]

        #Throw out any points that are 10* the median and recalculate
        #Do this until no points are removed.
        done = False
        while not done:
            if len(dat) == 0:
                break
            med = np.median(dat)
            fltr = [d for d in dat if d < 10*med]
            if len(fltr) == len(dat):
                done = True
            dat = fltr

        #now, retain the mean and stdev for later use
        if len(dat) > 2:
            avg.append(np.mean(dat))
            noise.append(np.std(dat))
        else:
            #something went wrong with noise detection... ignore section
            #probably a chunk containing only 0.0
            avg.append(0.0)
            noise.append(1.)

        if i + 1 == chunks:
            outnoise[i*chunk_size:] = noise[i]
            outbaseline[i*chunk_size:] = avg[i]
        else:
            outnoise[i*chunk_size:(i+1)*chunk_size] = noise[i]
            outbaseline[i*chunk_size:(i+1)*chunk_size] = avg[i]



    outx = []
    outy = []
    outidx = []
    outsnr = []

    #if not self.quiet:
    #    print("Locating peaks...")
    #if a point has SNR > threshold, look for local min in 2nd deriv.
    for i in range(2,len(yarray)-2):
        try:
            snr_i = (yarray[i] - avg[i // chunk_size])/noise[i // chunk_size]
        except IndexError:
            print("Cannot calculate SNR at spectrum index " + str(i))
            pass
        if snr_i >= snr:
            if (d2y[i-2] > d2y[i-1] > d2y[i] < d2y[i+1] or
                d2y[i-1] > d2y[i] < d2y[i+1] < d2y[i+2]):
                    outx.append(xarray[i])
                    outy.append(yarray[i])
                    outidx.append(i)
                    outsnr.append(snr_i)

    return outx, outy, outidx, outsnr, outnoise, outbaseline

if __name__ == "__main__":
    """ Takes a filepath to blackchirp output and performs peak finding based on
        Kyle's algorithm above.

        Input: Filename without extension of the file

        Optional: The SNR threshold, and the Delimiter

        Delimiter must be supplied as a explicit string:

        python peak_find.py ftb1870 2.0 "\t"
    """

    if len(sys.argv) < 2:
        print("Usage: python peak_find.py ftbfile [SNR threshold] [Delimiter]")
        raise TypeError("No blackchirp file specified")
    filepath = str(sys.argv[1]) + ".txt"
    if os.path.isfile(filepath) is False:
        raise FileNotFoundError(filepath + " does not exist. Try again!")
    try:
        # If SNR is provided
        snr = float(sys.argv[2])
    except IndexError:
        # Set a default value for the SNR
        snr = 4.
    try:
        delimiter = str(sys.argv[3])
        delim_whitespace=False
    except IndexError:
        delimiter = None
        delim_whitespace=True

    try:
        # This is done because Pandas is much more efficient at parsing
        # and writing data than NumPy. If it's available on the computer
        # this script is run on, we will use pandas instead of numpy.
        import pandas as pd
        pd_avail = True
    except ImportError:
        pd_avail = False
        pass
    """ Parse the blackchirp data """
    if pd_avail is True:
        # If Pandas is available, we'll use pandas instead (it's worth it)
        data = pd.read_csv(filepath,
                           delimiter=delimiter,
                           delim_whitespace=delim_whitespace,
                           header=None,
                           skiprows=1
                           )
        #data[data < 0.] = 0
        x = data[0].values
        y = data[1].values
    elif pd_avail is False:
        # If Pandas is not available, we'll use Numpy to parse and write
        fullarray = np.loadtxt(sys.argv[1], skiprows=1)
        x = fullarray[:,0]
        y = fullarray[:,1]
    outx, outy, outidx, outsnr, outnoise, outbaseline = find_peaks(x,y,snr)
    print(str(len(outy)) + " peaks found.")
    """ Write the peaks to a file """
    if pd_avail is True:
        peaks_df = pd.DataFrame(list(zip(outx, outy, outsnr)),
                                columns=["Frequency (MHz)",
                                         "Intensity",
                                         "SNR"
                                         ]
                                )
        peaks_df.sort_values("Intensity", ascending=False, inplace=True)
        peaks_df.to_csv(str(sys.argv[1]) + "_peaks.csv", index=False)
    elif pd_avail is False:
        np.savetxt(
            str(sys.argv[1]) + "_peaks.txt",
            np.array([outx, outy, outsnr]),
            delimiter='\t',
            header="Frequency (MHz)\tIntensity\tSNR"
                   )
    """ Plot the spectrum with peaks """
    fig, ax = plt.subplots()

    ax.plot(x, y, label="Data", lw=0.5)
    ax.vlines(outx, 0., outy, label="Peaks", color="#fec44f")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Intensity ($\mu$V)")
    ax.legend();

    fig.savefig(str(sys.argv[1]) + "_peaks.pdf", format="pdf")
