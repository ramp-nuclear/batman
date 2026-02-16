"""Tools for matrix exponentiation.

IPFCram16 is less accurate but faster. We don't seem to have a bottleneck in
the CRAM solver, so currently the solver of choice is IPFCram48.

"""

from .cram import *
from .protocol import Exponentiator, FluxFunc


IPFCram16 = IPFCramSolver(*cram16_coefficients())
IPFCram48 = IPFCramSolver(*cram48_coefficients())
Magnus = CF3Magnus2IPFCRAM(*cram48_coefficients())
