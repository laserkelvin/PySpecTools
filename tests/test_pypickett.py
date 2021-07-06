
from tempfile import NamedTemporaryFile
from pyspectools import pypickett


def test_linear():
    molecule = pypickett.LinearMolecule(
        None, **{"B": 102041.515, "D": 5.231e-3,}
    )
    # now test with hyperfine splitting
    molecule = pypickett.LinearMolecule(
        None, **{"B": 121515.241, "D": 1.162e-5, "H": 2.210, "chi_aa_1": -4.23, "chi_bb_2": -31.24}
    )
    assert molecule.num_nuclei == 2


def test_asymtop():
    molecule = pypickett.AsymmetricTop(**{"A": 10215., "B": 6024., "C": 2035.24, "DJ": 1e-3, "DJK": 1e-5})
    test = """         10000            1.021500e+04    0.000000e+00 /A
         20000            6.024000e+03    0.000000e+00 /B
         30000            2.035240e+03    0.000000e+00 /C
           200            1.000000e-03    0.000000e+00 /DJ
          1100            1.000000e-05    0.000000e+00 /DJK"""
    assert test.strip() == str(molecule).strip()
    # now test hyperfine splitting
    molecule = pypickett.AsymmetricTop(**{"A": 10215., "B": 6024., "C": 2035.24, "chi_aa_1": -402.5, "chi_bb_1": -214.51})
    test = """         10000            1.021500e+04    0.000000e+00 /A
         20000            6.024000e+03    0.000000e+00 /B
         30000            2.035240e+03    0.000000e+00 /C
     110010000           -4.025000e+02    0.000000e+00 /chi_aa_1
     110020000           -2.145100e+02    0.000000e+00 /chi_bb_1"""
    assert test.strip() == str(molecule).strip()


def test_generate_spcat():
    molecule = pypickett.AsymmetricTop(A=21516.262, B=2162.26, C=1268.6236)
    spcat = pypickett.SPCAT(prolate=False)
    par = spcat.format_var(molecule)
    static_par = """PySpecTools SPCAT input
 100  255    1    0    0.0000E+000    1.0000E+003    1.0000E+000 1.0000000000
s   1    -1    0   100    0    1    1    1     0   1   0
         10000            2.151626e+04    0.000000e+00 /A
         20000            2.162260e+03    0.000000e+00 /B
         30000            1.268624e+03    0.000000e+00 /C"""
    assert par.strip() == static_par.strip()
    int_file = spcat.format_int()
    static_int = """PySpecTools SPCAT input
 0  42   2.2415   0   99  -20.0  -5.0   300.0000  300.00
1  1.0002  0.0003  0.000"""
    assert static_int.strip() == int_file.strip()


def test_run_spcat():
    molecule = pypickett.AsymmetricTop(A=21516.262, B=2162.26, C=1862.6236)
    spcat = pypickett.SPCAT(T=30., prolate=True)
    initial_q, q_array = spcat.run(molecule, debug=True)