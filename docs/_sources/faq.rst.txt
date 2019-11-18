
PySpecTools FAQ
===============

This lists some of the most commonly used functionality, and most easily
forgotten functions accessible to PySpecTools.

Creating an experiment
~~~~~~~~~~~~~~~~~~~~~~

There are two main ways to create an `AssignmentSession` object, by creating the
class directly, or by using the `AssignmentSession.from_ascii()` function.

The latter method is a higher-level option, and allows spectra to be read in directly
from a variety of file formats. The former requires you to pass the spectrum as a
pandas `DataFrame` as a positional argument.

Creating an FTB file
~~~~~~~~~~~~~~~~~~~~

You can create FTB files (batch files used by QtFTM) from two sources, depending on
the type (either u-lines, or an existing catalog file) both of which wrap around the 
`LineList.to_ftb` method.

For u-lines, I recommend using the `AssignmentSession.create_uline_ftb_batch`, or
if you want to run a DR batch, `AssignmentSession.create_uline_dr_batch`. In both
cases, there are a few keyword arguments that allow you to specify the conditions
to carry out the batch, for example, the number of shots (`shots`), the dipole moment
used for the RF attenuation (`dipole`), and the minimum distance between lines (`gap`).

For other FTB files, say from a `.cat` or `.lin` file, you will want to create `LineList`
objects first and then use the `LineList.to_ftb()`.

