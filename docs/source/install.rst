Installation
============

It is recommended to use the provided ``conda.yml`` in the root directory to
set up a clean Python environment. This ensures that all the packages are
built and linked in the way that your platform expects prior to installing
``PySpecTools``, and create a ``conda`` environment called ``pst``.

OS-specific instructions
########################

Linux and Mac
*************

The following instructions assume you will be working in terminal, which is the
preferred way of installing things on Linux/Mac.

1. Download and install ``conda`` through the standard download; make sure you install Python 3.
2. Clone the ``PySpecTools`` git repository:
::

   git clone https://github.com/laserkelvin/PySpecTools.git
   cd PySpecTools
   conda env create -f conda.yml

Windows
*******

Installation on Windows is trickier for two reasons: PyTorch and Cython do not
play nearly as nicely as they should.

For Windows machines, you will need to include Visual Studio C++ libraries, as
they are needed by Cython for compilation. Installation for Linux systems
should automatically include the math library ``m``, whereas Mac OS this is not
required.


1. Download and install Anaconda through the usual means; make sure you install Python 3.
2. Download the github repository as a ZIP (through the Clone tab), and unpack it somewhere you'll find.
3. Open Anaconda Navigator in your start menu, and navigate to "Environments" to find the screen below

.. image:: https://docs.anaconda.com/_images/nav-env-tab.png
   :width: 700

4. Click on "Import" at the bottom left, and direct it to the ``conda.yml`` provided in the ``PySpecTools`` directory. This should build the ``conda`` environment named ``pst``.
5. Open an Anaconda command line through your start menu, and navigate to the ``PySpecTools`` directory.

All platforms
*************

1. When ``conda`` has finished working its magic, you can then proceed to install PySpecTools with ``pip``:
::

   pip install .

2. When updating, you can also use the github repository directly:
::

   pip install -U git+https://github.com/laserkelvin/PySpecTools

or by running ``git pull`` followed by ``pip install -U .``

Jupyter notebooks
#################

While not a requirement, much of the analysis workflow was designed with Jupyter
notebooks as a front-end for interactivity. Most base anaconda distributions
should already include Jupyter notebook in the installation, but if it does not
you can request it by running ``conda install jupyter``.

The default ``conda`` environment is ``base``, and typically this is the
environment you will be running Jupyter notebooks in without much thought.
To make sure the IPython kernel installed in the ``pst`` environment is
included in your ``base`` installation of Jupyter notebooks/lab, you will
need to follow the instructions found `here <https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove>`_.

Final notes
###########

You can test to make sure ``PySpecTools`` is installed correctly either by running the provided
tests, or at a lower level by trying to import ``PySpecTools` in a Python session.
