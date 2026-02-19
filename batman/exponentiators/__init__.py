"""Tools for matrix exponentiation.

IPFCram16 is less accurate but faster. We don't seem to have a bottleneck in
the CRAM solver, so currently the solver of choice is IPFCram48.

Examples
--------
>>> import scipy.sparse as sp
>>> import numpy as np
>>> d = sp.csr_matrix(np.array([[-np.log(2), 0.], [0., -np.log(2)]]))
>>> r = sp.csr_matrix(np.zeros((2,2)))
>>> f = lambda x: 0.
>>> n0 = np.ones(2)
>>> dt = 1
>>> exp: IPFCramSolver = IPFCram16
>>> bool(np.isclose(exp(d, r, f, n0, dt)[0], 0.5))
True

"""

from batman.exponentiators.cram import *
from batman.exponentiators.protocol import Exponentiator, FluxFunc


IPFCram16 = IPFCramSolver(*cram16_coefficients())
IPFCram48 = IPFCramSolver(*cram48_coefficients())
Magnus = CF3Magnus2IPFCRAM(*cram48_coefficients())
