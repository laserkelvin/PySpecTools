from pyspectools import routines

AlS = routines.pickett_molecule("AlS.json")

AlS.setup_par()
AlS.setup_int()
AlS.predict_lines()

