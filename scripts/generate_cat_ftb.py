#!/usr/bin/env python

from pyspectools import ftmw_analysis as fa
import pandas as pd
import click

@click.command()
@click.argument(
    "filepath",
    type=click.Path(exists=True)
    )
@click.argument(
    "cat_type",
    type=click.Choice(
        ["dipole", "magnet", "DR", "discharge", "atten"]
        )
    )
@click.option(
    "--intensities",
    default=False,
    is_flag=True,
    help="If True, intensity column will be used in the text file."
    )
@click.option(
    "--attn",
    default=False,
    is_flag=True,
    help="Use attenuations in the file instead of dipole moments."
    )
def run_gencatftb(filepath, cat_type, intensities=False, attn=False):
    """ generate_cat_ftb.py

        This script will take a file containing frequencies and
        possibly intensities and generate a FT batch file.

        The available categories are:
        Dipole tests
        Attenuation tests
        Magnet tests
        Double-resonance tests
        Discharge tests

        Please only request one test at a time.
    """
    delimiter = input("Please type the delimiter character in the file:    ")

    dataframe = pd.read_csv(filepath, delimiter=delimiter, header=None, comment="#")

    print("First five lines of data readin - check delimiting!")
    print(dataframe.head())

    options = dict()

    print("You chose to perform a " + cat_type + " test.")

    if cat_type == "dipole":
        options["dipole"] = True

    if cat_type == "atten":
        options["atten"] = True
        
    elif attn is True:
        attn_col = int(input("Please specify the column number for attenuation:    "))
        attn = dataframe[attn_col]

    elif cat_type == "magnet":
        options["magnet"] = True

    elif cat_type == "DR":
        options["dr"] = True

    elif cat_type == "discharge":
        options["discharge"] = True

    if intensities is False:
        nshots = int(input("Please specify the number of shots on each line:    "))
        options["nshots"] = nshots
    else:
        int_col = int(input("Please specify the column number for intensities:    "))
        intensities = dataframe[int_col]
        options["intensities"] = intensities

    ftb_line = fa.categorize_frequencies(
        dataframe[0],
        **options
        )

    filename = input("Please specify name for file; .ftb will be added automatically.   ")
    with open(filename + ".ftb", "w+") as write_file:
        write_file.write(ftb_line)


if __name__ == "__main__":
    run_gencatftb()
