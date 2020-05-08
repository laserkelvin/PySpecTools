
PySpecTools FAQ
===============

This lists some of the most commonly used functionality, and most easily
forgotten functions accessible to PySpecTools.

Creating an experiment
~~~~~~~~~~~~~~~~~~~~~~

There are two main ways to create an ``AssignmentSession`` object, by creating
the class directly, or by using the ``AssignmentSession.from_ascii()``
function.

The latter method is a higher-level option, and allows spectra to be read in
directly from a variety of file formats. The former requires you to pass the
spectrum as a pandas ``DataFrame`` as a positional argument.

Creating an FTB file
~~~~~~~~~~~~~~~~~~~~

You can create FTB files (batch files used by QtFTM) from two sources,
depending on the type (either u-lines, or an existing catalog file) both of
which wrap around the ``LineList.to_ftb`` method.

For u-lines, I recommend using the
``AssignmentSession.create_uline_ftb_batch``, or if you want to run a DR batch,
``AssignmentSession.create_uline_dr_batch``. In both cases, there are a few
keyword arguments that allow you to specify the conditions to carry out the
batch, for example, the number of shots (``shots``), the dipole moment used for
the RF attenuation (``dipole``), and the minimum distance between lines
(``gap``).

For other FTB files, say from a ``.cat`` or ``.lin`` file, you will want to
create ``LineList`` objects first and then use the ``LineList.to_ftb()``.

Instrumental RFI removal
~~~~~~~~~~~~~~~~~~~~~~~~

Strong RFI are common in broadband spectra, which arise from instrument clocks
and other electronic components. The ones that are most prevalent for us in the
McCarthy group are typically identified by two characteristics: often razor
sharp, and many of them occurring at near-integer frequencies that are unlikely
molecular in origin (e.g. 16500.0000 MHz). ``PySpecTools`` has three approaches
to managing these instrumental artifacts, the simplest being an automatic
removal that looks for near integer frequencies
(``pyspectools.spectra.analysis.detect_artifacts`` function).

The next level of complexity is based on calculating sub-harmonics, and
sum/difference-frequency combinations of sub-harmonics, based on some clock
frequency. For our Keysight arbitrary waveform generator, we've found the most
prominent artifacts are derived from the 65 GHz clock. The
``LineList.from_clock`` method was created for this purpose, which generates
multiples of this clock frequency, and calculates every possible sum/difference
combination of pairs based on this list.
``AssignmentSession.process_clock_spurs`` provides an interface to this
function, which will automatically assign features found this way - it is
recommended to run this function at the end of analysis once molecular features
are exhausted, as these clock frequencies can be incredibly dense.

Finally, the most manual method is to create a artifact ``LineList`` object via
the ``LineList.from_artifacts`` method. This takes a list of frequencies that
the user specifies, and uses these to assign features as artifacts.

Report generation
~~~~~~~~~~~~~~~~~

Once you've done all the analysis, the fun part begins: writing the damn paper.
The ``AssignmentSession.finalize_assignments`` function is not as final as it
sounds, and the current behavior is to generate an HTML report for sharing, as
well as an overview. This function will also generate a LaTeX table in the
``reports`` folder, which summarizes the findings of your experiment.

Currently, ``PySpecTools`` does not automatically write the paper for you.

