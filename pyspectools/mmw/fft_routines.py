
import numpy as np
from numpy.fft import fft, ifft
from scipy import signal as spsig

"""
    Fourier filter
"""

def fft_filter(ydata, window_function=None, cutoff=[50, 690], sample_rate=None):
    """
        Fourier filter implementation.
        
        Takes signal data in the frequency domain, and transforms into the
        time domain where slices are taken off.
        
        Various window functions can be applied, including stock and custom
        filters that I have implemented. One in particular is the house
        filter, which is considered the "gold standard".
    """
    stock_windows = [
        "blackmanharris",
        "blackman",
        "boxcar",
        "gaussian",
        "hanning",
        "bartlett",
    ]

    custom_windows = [
        "lowpass",
        "highpass",
        "bandpass",
        "chebyshev",
        "house"
    ]
    # If a window function is provided, apply it to the frequency-domain signal
    # before applying the Fourier transform.
    if window_function:
        if window_function not in stock_windows and window_function not in custom_windows:
            pass
        if window_function in stock_windows:
            ydata *= spsig.get_window(window_function, ydata.size)
        else:
            if window_function != "chebyshev" and window_function != "house":
                    #raise InvalidBandPassError("No frequency cut off supplied.")
                ydata *= butter_filter(ydata, cutoff, sample_rate, window_function)
            elif window_function == "chebyshev":
                ydata *= spsig.chebwin(ydata.size, 200.)
    # FFT the frequency spectrum to time domain
    time_domain = fft(ydata)
    
    # If no range is specified, take a prespecified chunk
    if cutoff is None:
        cutoff = [50, 690]
    # Make sure values are sorted
    cutoff = sorted(cutoff)
    
    # Apply the house filter before performing the inverse
    # FT back to frequency domain.
    if window_function == "house":
        house_window = house_filter(time_domain.size, *cutoff)
        time_domain *= house_window
        time_domain[:min(cutoff)] = 0.
        time_domain[max(cutoff):] = 0.
    
    # Return the real part of the inverse FT
    filtered = np.real(ifft(time_domain))
    return filtered

"""
    Custom coded filters
"""

def gen_butter_filter(cutoff: np.ndarray, filter_type, order=5):
    """
    [summary]

    Parameters
    ----------
    cutoff : [type]
        [description]
    fs : [type]
        [description]
    filter_type : [type]
        [description]
    order : int, optional
        [description], by default 5

    Returns
    -------
    [type]
        [description]
    """
    # nyq = 0.5 * fs
    # normal_cutoff = [freq / nyq for freq in cutoff]
    b, a = spsig.butter(
        order, cutoff, btype=filter_type, analog=False
    )
    return b, a


def butter_filter(data, cutoff, fs, filter_type, order=5):
    # Generate filter window function
    b, a = gen_butter_filter(cutoff, fs, filter_type, order=order)
    y = spsig.lfilter(b, a, data)
    return y


def house_filter(size, low, high):
    """
        Function that returns the "gold standard" filter.
        
        This window is designed to produce low sidelobes
        for Fourier filters.
        
        In essence it resembles a sigmoid function that
        smoothly goes between zero and one, from short
        to long time.
    """
    filt = np.zeros(size)
    
    def eval_filter(rf, c1, c2, c3, c4):
        r1 = 1. - rf**2.
        r2 = r1**2.
        r3 = r2 * r1
        filt = c1 + c2*r1 + c3*r2 + c4*r3
        return filt

    coefficients = {
        "c1": 0.074,
        "c2": 0.302,
        "c3": 0.233,
        "c4": 0.390
    }
    
    denom = (high - low + 1.0) / 2.
    if denom < 0.:
        raise ZeroDivisionError
    
    for i in range(int(low), int(high)):
        rf = (i + 1) / denom
        if rf > 1.5:
            filt[i] = 1.
        else:
            temp = eval_filter(rf, **coefficients)
            if temp < 0.:
                filt[i] = 1.
            else:
                filt[i] = 1. - temp
    filt[int(high):] = 1.
    return filt