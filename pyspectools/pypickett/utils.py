
import contextlib
import os
from tempfile import TemporaryDirectory


par_template = """PySpecTools SPCAT input
 100  255    1    0    0.0000E+000    1.0000E+003    1.0000E+000 1.0000000000
{reduction}   {quanta}    {top}    0   {k_max}    0    {weight_axis}    {even_weight}    {odd_weight}     0   1   0
{parameters}
"""

int_template = """PySpecTools SPCAT input
 0  {mol_id}   {q:.4f}   0   {max_f_qno}  {int_min:.1f}  {int_max:.1f}   {freq_limit:.4f}  {T:.2f}
{dipole_moments}
"""

@contextlib.contextmanager
def work_in_temp():
    """
    Context manager for working in a temporary directory. This
    simply uses the `tempfile.TemporaryDirectory` to create and
    destroy the folder after using, and manages moving in and
    out of the folder.
    """
    cwd = os.getcwd()
    try:
        with TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            yield
    finally:
        os.chdir(cwd)