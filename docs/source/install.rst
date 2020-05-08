Installation
============

``PySpecTools`` can be installed with any version of Python 3.7+ simply with
``pip``.  After cloning the git repository, you can install it via command line
by navigating to where you cloned the repository, and running:

``pip install .``

Updates of the routines can be downloaded by running ``git pull`` in the top
directory, followed by ``pip install -U .``.

For Windows machines, you will need to include Visual Studio C++ libraries, as
they are needed by Cython for compilation. Installation for Linux systems
should automatically include the math library ``m``, whereas Mac OS this is not
required.

While not a requirement, much of the analysis workflow was designed with Jupyter
notebooks as a front-end for interactivity. Most base anaconda distributions
should already include Jupyter notebook in the installation, but if it does not
you can request it by running ``conda install jupyter``.


