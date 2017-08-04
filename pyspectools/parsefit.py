
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from plotly.tools import mpl_to_plotly
from plotly.offline import init_notebook_mode, iplot, enable_mpl_offline

class fit_output:
    def __init__(self, fit_file, verbose=True, interactive=False):
        # Check if the .fit file exists
        if os.path.isfile(fit_file) is False:
            raise FileNotFoundError(fit_file + " does not exist. No .fit file!")

        self.fit_properties = {
            "number of parameters": 0,
            "rms errors": list(),
            "fit rms": list(),
            "average errors": list(),
            "final rms": 0.,
            "parameter optimization": dict(),
            "line progress": dict(),
            "iteration count": 1,
            "fit iterations": 0,
            "fit file": fit_file,
            "linear": False
        }
        self.interactive = interactive
        self.data = dict()
        self.parsefit()
        self.analyze_parse(verbose)

    def parsefit(self):
        iteration_flag = False
        parameter_flag = False
        bad_line = False
        with open(self.fit_properties["fit file"], "r") as read_file:
            for line in read_file.readlines():
                if "LINEAR MOLECULE" in line:
                    self.fit_properties["linear"] = True
                if "NORMALIZED DIAGONAL" in line:
                    iteration_flag = False
                    self.fit_properties["line progress"][self.fit_properties["iteration count"]] = iteration_dict
                if "Fit Diverging" in line:
                    iteration_flag = False
                    self.fit_properties["line progress"][self.fit_properties["iteration count"]] = iteration_dict
                if "NEXT LINE NOT USED IN FIT" in line:
                    bad_line = True
                if iteration_flag is True:
                    split_line = line.split()
                    if len(split_line) > 7:
                        if len(split_line) == 8:
                            # Linear molecule with no hyperfine
                            iteration_dict[str(fit_line_count)] = {
                                "line number": fit_line_count,
                                "exp freq": float(split_line[3]),
                                "calc freq": float(split_line[4]),
                                "diff": float(split_line[5]),
                                "lower state": [
                                    int(split_line[2]),
                                ],
                                "upper state": [
                                    int(split_line[1])
                                ],
                                "uncertainty": float(split_line[6]),
                                "bad line": bad_line
                            }
                        else:
                            # Works for linear molecule with hyperfine
                            iteration_dict[str(fit_line_count)] = {
                                "line number": fit_line_count,
                                "exp freq": float(split_line[7]),
                                "calc freq": float(split_line[8]),
                                "diff": float(split_line[9]),
                                "lower state": [
                                    int(split_line[4]),
                                    int(split_line[5]),
                                    int(split_line[6])
                                ],
                                "upper state": [
                                    int(split_line[1]),
                                    int(split_line[2]),
                                    int(split_line[3])
                                ],
                                "uncertainty": float(split_line[10]),
                                "bad line": bad_line
                            }
                        fit_line_count += 1
                        bad_line = False
                if "EXP.FREQ." in line:
                    iteration_flag = True
                    fit_line_count = 0
                    iteration_dict = dict()
                if "MICROWAVE AVG" in line:
                    parameter_flag = False
                    self.fit_properties["parameter optimization"][self.fit_properties["iteration count"]] = parameter_dict
                    split_line = line.split()
                    self.fit_properties["average errors"].append(float(split_line[3]))
                if "MICROWAVE RMS" in line:
                    self.fit_properties["rms errors"].append(float(split_line[3]))
                if parameter_flag is True:
                    # Remove the brackets
                    for bracket in ['''(''',''')''']:
                        line = line.replace(bracket, " ")
                    split_line = line.split()
                    if len(split_line) == 6:
                        parameter_dict[split_line[2]] = {
                            "value": str(split_line[3]),
                            "uncertainty": str(split_line[4]),
                            "change": float(split_line[5])
                        }
                    else:
                        # Sometimes the exponent is truncated, and this case
                        # will extract the exponent as well
                        parameter_dict[split_line[2]] = {
                            "value": float(split_line[3] + split_line[5]),
                            "uncertainty": split_line[4],
                            "change": float(split_line[6])
                        }
                if "NEW PARAMETER" in line:
                    parameter_dict = dict()
                    parameter_flag = True
                if "END OF ITERATION" in line:
                    self.fit_properties["iteration count"] += 1
                    split_line = line.split()
                    self.fit_properties["fit rms"].append(float(split_line[8]))
                if "Bad Line" in line:
                    print(line)

    def analyze_parse(self, verbose=True):
        for iteration in self.fit_properties["line progress"]:
            self.data[iteration] = pd.DataFrame.from_dict(
                self.fit_properties["line progress"][iteration]
            ).T
        if "line number" in list(self.data[iteration].keys()):
            # This check will make sure there are lines in our parsed data
            if verbose is True:
                # In manual mode, we'll plot the errors and changes in the parameters
                self.plot_error(iteration)
                #self.parameter_changes()           # Not useful, so not plotting
            self.data[iteration].sort_values("line number", inplace=True)
            self.data[iteration].to_csv("exp-calc.csv")
            niterations = len(self.fit_properties["rms errors"])
            self.fit_properties["final rms"] = self.fit_properties["fit rms"][niterations - 1]
            print("Final RMS error:\t" + str(self.fit_properties["final rms"]))
            print("Microwave frequency RMS error:\t" + str(self.fit_properties["rms errors"][-1]))
        else:
            # If there are no lines, this "exception" will make sure that the
            # routine exits gracefully
            print("There are no lines fit in the final iteration!")
            print("No analysis was done on this iteration.")

    def export_parameters(self):
        last_iteration = self.fit_properties["iteration count"] - 1
        return self.fit_properties["parameter optimization"][last_iteration]

    def parameter_changes(self):
        """ Plot the change in the parameters - compare the first and last
            iterations of the fitting procedure.
        """
        differences = list()
        labels = list()
        for parameter in self.fit_properties["parameter optimization"][1]:
            try:
                differences.append(
                    float(self.fit_properties["parameter optimization"][1][parameter]["value"]) - \
                    float(self.fit_properties["parameter optimization"][self.fit_properties["iteration count"] - 1][parameter]["value"])
                )
                labels.append(parameter)
            except ValueError:
                print("Parse error on " + parameter)
                pass

        # save the change in parameter space as a dataframe
        self.change = [differences, labels]

        fig, ax = plt.subplots(figsize=(14, 5.5))

        ax.set_title("Overall change in parameters ")
        ax.set_xticks(np.arange(len(differences)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Overall change (MHz)")

        fig.savefig("parameter_change_plot.pdf", format="pdf")

    def plot_error(self, iteration):
        """ Plots the observation - calculated deviation, and the overall
            fitting RMS as a function of iteration count, for an arbitrary
            iteration number.
        """
        fig, axarray = plt.subplots(1, 2, figsize=(14,5.5))
        axarray[0].errorbar(
                   x=self.data[iteration]["line number"].astype(float),
                   y=self.data[iteration]["diff"].astype(float),
                   yerr=self.data[iteration]["uncertainty"].astype(float),
                   fmt="o",
                   color="#fec44f",
                   ls="none",
                   alpha=0.7,
                   label="Fitted frequencies"
                  )
        bad_data = self.data[iteration].loc[self.data[iteration]["bad line"] == True]
        axarray[0].scatter(
            x=bad_data["line number"].astype(float),
            y=bad_data["diff"].astype(float),
            color="#f03b20",
            label="Bad frequencies",
            marker="D"
        )
        axarray[0].axhline(y=0, linewidth=2, color="#31a354", alpha=0.7)
        axarray[0].set_title("Line deviation for iteration " + str(iteration))
        axarray[0].set_xlabel("Line number")
        axarray[0].set_ylabel("Exp. - Calc. frequency (MHz)")
        axarray[0].legend()

        axarray[1].scatter(
            np.arange(self.fit_properties["iteration count"] - 1,),
            self.fit_properties["fit rms"],
            color="#f03b20"
        )
        axarray[1].set_title("RMS error summary plot")
        axarray[1].set_xlabel("Iteration count")
        axarray[1].set_xticks(np.arange(self.fit_properties["iteration count"] - 1))
        axarray[1].set_ylabel("Microwave RMS error (MHz)")

        fig.savefig("rms_plot_" + str(iteration) + ".pdf", format="pdf")

        if self.interactive is True:
            plt.close(fig)
            pltly_fig = mpl_to_plotly(fig)
            try:
                iplot(pltly_fig)
            except PlotlyEmptyDataError:
                pass
