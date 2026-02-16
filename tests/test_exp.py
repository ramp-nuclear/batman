"""Test matrix exponentiation schemes.

"""

import hypothesis.strategies as st
import numpy as np
import pytest
import scipy.sparse.linalg as spla
from hypothesis import given, settings
from scipy.sparse import csr_matrix

from batman.exponentiators import IPFCramSolver, \
    Exponentiator, CF3Magnus2IPFCRAM as Magnus
# noinspection PyPep8Naming
from batman.exponentiators import cram16_coefficients as CRAM16
# noinspection PyPep8Naming
from batman.exponentiators import cram48_coefficients as CRAM48


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture(scope='module')
def cram16_serial() -> Exponentiator:
    return IPFCramSolver(*CRAM16())


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture(scope='module')
def cram48_serial() -> Exponentiator:
    return IPFCramSolver(*CRAM48())


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture(scope='module')
def magnus48() -> Exponentiator:
    return Magnus(*CRAM48())


@pytest.mark.parametrize(('m', 'v', 't', 'res'),
                         [(csr_matrix([[1., 0.], [0., 1.]]),
                           np.array([5., 4.]), 0.,
                           np.array([5., 4.])),
                          (csr_matrix([[1., 0.], [0., 1.]]),
                           np.array([5., 4.]), 1.,
                           np.array([5.*np.e, 4.*np.e]))
                          ])
def test_cram_16_on_identity_matrix(
        cram16_serial: Exponentiator,
        m: csr_matrix,
        v: np.ndarray,
        t: float,
        res: np.ndarray):
    z = csr_matrix(m.shape)
    assert np.allclose(cram16_serial(m, z, lambda x: 0., v, t), res)


@pytest.mark.parametrize(('m', 'v', 't', 'res'),
                         [(csr_matrix([[1., 0.], [0., 1.]]),
                           np.array([5., 4.]), 0.,
                           np.array([5., 4.])),
                          (csr_matrix([[1., 0.], [0., 1.]]),
                           np.array([5., 4.]), 1.,
                           np.array([5.*np.e, 4.*np.e]))
                          ])
def test_cram_48_on_identity_matrix(
        cram48_serial: Exponentiator,
        m: csr_matrix,
        v: np.ndarray,
        t: float,
        res: np.ndarray):
    z = csr_matrix(m.shape)
    assert np.allclose(cram48_serial(m, z, lambda x: 0., v, t), res)


non_zero = (st.floats(min_value=1e-3, max_value=1.,
                      allow_infinity=False, allow_nan=False)
            | st.floats(max_value=-1e-3, min_value=-1.,
                        allow_infinity=False, allow_nan=False)
            )
vec_strat = st.tuples(non_zero, non_zero)
mat_strat = st.lists(elements=vec_strat, min_size=2, max_size=2, unique=True)
time_strat = st.floats(min_value=0.5, max_value=3.)


@settings(max_examples=100)
@given(m=mat_strat, v=vec_strat, t=time_strat)
def test_cram48_vs_pade_on_identity_matrix(
        cram48_serial: Exponentiator,
        m: csr_matrix,
        v: np.ndarray,
        t: float):
    m = csr_matrix(m)
    z = csr_matrix(m.shape)
    vec = np.array(v)
    cramsol = cram48_serial(m, z, lambda x: 0., vec.copy(), t)
    padesol = spla.expm_multiply(m, vec, 0., t,
                                 num=5, endpoint=True)[-1]
    assert np.allclose(cramsol, padesol, rtol=1e-3)


@settings(max_examples=200)
@given(d=mat_strat, r=mat_strat, n=vec_strat, t=time_strat)
def test_cram_is_same_as_magnus_for_constant_flux(
        cram48_serial: Exponentiator, magnus48: Exponentiator,
        d: csr_matrix, r: csr_matrix,
        n: np.array, t: float):
    d, r = csr_matrix(d), csr_matrix(r)
    n = np.array(n)
    cramsol = cram48_serial(d, r, lambda x: 1., n, t)
    magnussol = magnus48(d, r, lambda x: 1., n, t)
    assert np.allclose(cramsol, magnussol)
