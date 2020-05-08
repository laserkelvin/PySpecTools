"""
    formatting.py

    Routines for formatting output/data into publication
    ready tables and whatnot.
"""


def read_lin(filepath, labels):
    """ Function that will read in a .lin file and quantum number
        labeling, and return a formatted LaTeX table.
        
        The labels should be supplied without indication of
        upper or lower state, i.e. J and not J''.
        
        Requires:
        ------------
        filepath - path to the .lin file
        labels - tuple-like, lists the quantum number labels
    """
    df = pd.read_csv(filepath, delim_whitespace=True, header=None)
    grouped_labels, upper_labels, lower_labels = generate_labels(labels)
    # Tack on the remaining three columns after the quantum numbers
    col_headings = upper_labels + lower_labels
    col_headings.extend(["Frequency", "Uncertainty", "_"])
    df.columns = col_headings
    df = df[grouped_labels + ["Frequency", "Uncertainty", "_"]]
    hyperfine_labels = [value for value in df.keys() if "F" in value]
    # Get rid of frequency
    hyperfine_labels.remove("Frequency")
    if len(hyperfine_labels) > 0:
        df[hyperfine_labels] -= 0.5
    return df


def generate_labels(labels):
    """ Function that will generate the quantum number
        labelling for upper and lower states.
        
        Requires:
        ------------
        labels - tuple-like containing quantum number labels
        without indication of upper or lower state
        
        Returns:
        ------------
        grouped, upper, lower - lists corresponding to the
        labels groups by quantum number (rather than upper/lower),
        and the upper and lower state quantum numbers respectively.
    """
    grouped_labels = list()
    lower_labels = list()
    upper_labels = list()
    for label in labels:
        for index, symbol in enumerate(["""'""", """''"""]):
            header = label + symbol
            grouped_labels.append(header)
            if index == 0:
                upper_labels.append(header)
            else:
                lower_labels.append(header)
    return grouped_labels, upper_labels, lower_labels


def lin2latex(dataframe, labels=None, deluxe=False):
    """ Function that will convert a dataframe into a LaTeX
        formatted table. Optionally, you can select specific
        quantum numbers to include in the table.
        
        The formatting follows the "house style",
        which groups rows by J, followed by N and F.
        
        The labels should be supplied without indication of
        upper or lower state, i.e. J and not J''. Also, labels
        should be in the order you want them!!!
        
        Requires:
        ------------
        dataframe - dataframe read in the format from `read_lin`
        labels - optional tuple-like which lists out which quantum
        numbers to include. Defaults to printing all the quantum
        numbers.
    """
    if labels is None:
        # Take all the keys except the weird column
        columns = [key for key in dataframe.keys() if key != "_"]
    else:
        # Generate the labeling
        group, upper, lower = generate_labels(labels)
        columns = group
        columns.extend(["Frequency", "Uncertainty"])
    # Template for a deluxe table used by ApJ and others
    if deluxe is True:
        template = """
        \\begin{{deluxetable}}{colformat}
            \\tablecaption{{Something intelligent}}
            \\tablehead{{header}}
            
            \\startdata
                {data}
            \\enddata
        \\end{{deluxetable}}
        """
    # Template for a normal LaTeX table
    elif deluxe is False:
        template = """
        \\begin{{table}}
            \\begin{{center}}
                \\caption{{Something intelligent}}
                \\begin{{tabular}}{colformat}
                {header}
                \\toprule
                
                {data}
                \\bottomrule
                \\end{{tabular}}
            \\end{{center}}
        \\end{{table}}
        """
    data_str = ""
    # Take only what we want
    filtered_df = df[columns]
    # Sort the dataframe by J
    filtered_df.sort_values(["J'", "J''"])
    for index, row in filtered_df.iterrows():
        values = list(row.astype(str))
        # We want to skip printing J transitions if they're the same
        if index == 0:
            Jlast = values[:2]
        else:
            Jcurr = values[:2]
            # If the current values of J match the previous, then
            # set them to blank so we don't print them
            if Jcurr == Jlast:
                values[0] = " "
                values[1] = " "
            Jlast = Jcurr
        data_str += " & ".join(values) + "\\\\\n"
    # swap this out if you want columns to not be centered
    column_format = ["c"] * len(columns)
    column_format = "{" + " ".join(column_format) + "}"
    data_dict = {
        "colformat": column_format,  # centered
        "data": data_str,
        "header": " & ".join(columns) + "\\\\",
    }
    return template.format_map(data_dict)
