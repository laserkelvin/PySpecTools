import pandas as pd
import numpy as np
import sys

def pick_pickett(simulation_path, low_freq=0., high_freq=np.inf, threshold=-np.inf):
    """ Parses a simulation output, and filters the frequency and intensity to give
    a specific set of lines.
    The only argument that is required is the path to the simulation output. Others
    are optional, and will default to effectively not filter.
    """
    clean_cat(simulation_path)
    simulation_df = pd.read_csv(simulation_path, delim_whitespace=True, header=None)
    thresholded_df = simulation_df.loc[
            (simulation_df[0].astype(float) >= low_freq) &         # threshold the simulation output
            (simulation_df[0].astype(float) <= high_freq) &        # based on user specified values
            (simulation_df[2].astype(float) >= threshold)          # or lack thereof
            ]
    thresholded_df.rename(
            columns={
                0: "Frequency",
                1: "Uncertainty",
                2: "Intensity",
                3: "DoF",
                4: "Lower state energy",
                5: "Degerency",
                6: "Quantum number coding",       # trying to label the columns
                },
            inplace=True
            )
# This bit combines the three columns which are meant to be for the quantum numbers
# The output from Pickett does not format the quantum numbers well for parsing, sometimes
# the numbers merge (i.e. if '2 4 10' might be written as '2 410'
    if len(thresholded_df.keys()) >= 12:
        thresholded_df["Lower quantum numbers"] = thresholded_df[[8, 9, 10]].apply(lambda x: " ".join(map(str,x)), axis=1)
        thresholded_df["Upper quantum numbers"] = thresholded_df[[11, 12, 13]].apply(lambda x: " ".join(map(str,x)), axis=1)
        #thresholded_df["Lower quantum numbers"] = thresholded_df.loc[:, [8, 9, 10]]#.apply(join_function)
        #thresholded_df["Upper quantum numbers"] = thresholded_df.loc[:, [11, 12, 13]]#.apply(join_function)
    else:
        pass
    return thresholded_df


def clean_cat(filepath):
    """ This function will clean up a .cat file; due to the formatting
        nightmare that is SPCAT, I have to resort to cheap and dirty ways
        of fixing the uncertainties that get spit out.
    """
    with open(filepath) as read_file:
        cat_contents = read_file.read()

    cat_contents = cat_contents.replace("999.9999", "0.0000".rjust(8))

    with open(filepath, "w+") as write_file:
        write_file.write(cat_contents)

    print("Cleaned up the .cat")

def extract_experimental_lines(thresholded_df):
    """ Lines that are experimental are denoted by a negative sign. This
        function just checks the sign, and returns two dataframes that correspond
        to the experimental and theoretical lines respectively.
    """
    exp_df = thresholded_df[thresholded_df[6] < 0]
    predicted_df = thresholded_df[thresholded_df[6] > 0]
    return exp_df, predicted_df


def peak2cat(peaks_df, outputname="generated_batch.ftb", sortby="Intensity", ascending=True, nshots=100):
    if sortby not in ["Intensity", "Frequency"]:
        print("You can't sort by " + sortby)
        print("Choose 'Intensity' or 'Frequency', case sensitive.")
        sys.exit()
    sorted_df = peaks_df.sort_values(
            by=sortby,
            ascending=False,
            inplace=True
            )
    frequencies = peaks_df["Frequency"].values
    intensities = peaks_df["Intensity"].values
    """ Now we convert the relative intensities into a number of shots """
    intensities = (10.**intensities)
    intensities = intensities / intensities.max()    # relative to the strongest
    shotcounts = intensities * nshots                # scale shots
    if "Lower quantum numbers" and "Upper quantum numbers" in peaks_df.keys():
        lower_num = peaks_df["Lower quantum numbers"].values
        upper_num = peaks_df["Upper quantum numbers"].values
        fullarray = np.concatenate(
                (
                frequencies,
                shotcounts,
                lower_num,
                upper_num
                )
             )
    else:
        """ If we can't read the quantum numbers, pad the columns with zeros """
        fullarray = np.concatenate(
                (
                frequencies,
                shotcounts,
                np.zeros(frequencies.size),
                np.zeros(frequencies.size)
                )
            )
    fullarray = fullarray.reshape(4, frequencies.size)
    fullarray = fullarray.T                              # transpose into colums
    np.savetxt(
            fname=outputname,
            X=fullarray,
            fmt=("ftm:%5.3f", " shots:%1i", "# lower:%5s", " upper:%5s")
            )


def join_function(x):
    # function to join quantum numbers together
    return " ".join(str(x))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pick_pickett.py [simulation output]")
        print("Optional arguments:\t [low] and [high] frequency thresholds, [intensity] threshold")
        sys.exit()

    simulation_path = str(sys.argv[1])
    parameters = [float(value) for value in sys.argv[2:]]
    print(parameters)
    # threshold the simulation values
    thresholded_df = pick_pickett(simulation_path, *parameters)
    # export the thresholded values to a cat file
    thresholded_df.to_csv(
            simulation_path.split(".")[0] + "_filtered.cat",
            sep=" ",
            header=None,
            index=False
            )
    # specify user input, and use default values if passed
    outputname = input("Specify a name for the batch file (default: generated_batch.ftb)")
    if outputname == "":
        outputname = "generated_batch.ftb"
    sortby = input("What do you want to sort by: Intensities or Frequency (default: Frequency)")
    if sortby == "":
        sortby = "Frequency"
    direction = input("Ascending order? Y/N (default: Y)")
    if direction != "Y" or direction != "y":
        ascending = False
    else:
        ascending = True
    shotcount = input("Number of shots at strongest line? (default: 100)")
    if shotcount == "":
        shotcount = 100
    else:
        shotcount = int(shotcount)
    # call the function
    peak2cat(
            peaks_df=thresholded_df,
            outputname=outputname,
            sortby=sortby,
            ascending=ascending,
            nshots=shotcount
            )
