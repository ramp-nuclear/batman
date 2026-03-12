"""Test matrix exponentiation schemes."""

import hypothesis.strategies as st
import numpy as np
import pytest
import scipy.sparse.linalg as spla
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from scipy.sparse import csr_matrix

from batman.exponentiators import CF3Magnus2IPFCRAM as Magnus
from batman.exponentiators import Exponentiator, IPFCramSolver
from batman.exponentiators import cram16_coefficients as CRAM16
from batman.exponentiators import cram48_coefficients as CRAM48


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture(scope="module")
def cram16_serial() -> Exponentiator:
    return IPFCramSolver(*CRAM16())


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture(scope="module")
def cram48_serial() -> Exponentiator:
    return IPFCramSolver(*CRAM48())


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture(scope="module")
def magnus48() -> Exponentiator:
    return Magnus(*CRAM48())


@pytest.mark.parametrize(
    ("m", "v", "t", "res"),
    [
        (csr_matrix([[1.0, 0.0], [0.0, 1.0]]), np.array([5.0, 4.0]), 0.0, np.array([5.0, 4.0])),
        (csr_matrix([[1.0, 0.0], [0.0, 1.0]]), np.array([5.0, 4.0]), 1.0, np.array([5.0 * np.e, 4.0 * np.e])),
    ],
)
def test_cram_16_on_identity_matrix(
    cram16_serial: Exponentiator, m: csr_matrix, v: np.ndarray, t: float, res: np.ndarray
):
    z = csr_matrix(m.shape)
    assert np.allclose(cram16_serial(m, z, lambda x: 0.0, v, t), res)


@pytest.mark.parametrize(
    ("m", "v", "t", "res"),
    [
        (csr_matrix([[1.0, 0.0], [0.0, 1.0]]), np.array([5.0, 4.0]), 0.0, np.array([5.0, 4.0])),
        (csr_matrix([[1.0, 0.0], [0.0, 1.0]]), np.array([5.0, 4.0]), 1.0, np.array([5.0 * np.e, 4.0 * np.e])),
    ],
)
def test_cram_48_on_identity_matrix(
    cram48_serial: Exponentiator, m: csr_matrix, v: np.ndarray, t: float, res: np.ndarray
):
    z = csr_matrix(m.shape)
    assert np.allclose(cram48_serial(m, z, lambda x: 0.0, v, t), res)


def minus_major(m: np.ndarray) -> csr_matrix:
    m[0, 0] = -abs(m[0, 1]) - abs(m[0, 0])
    m[1, 1] = -abs(m[1, 0]) - abs(m[1, 1])
    return csr_matrix(m)


positives = st.floats(min_value=0, max_value=10, allow_subnormal=False)
mvalues = st.floats(min_value=-10, max_value=10, allow_subnormal=False)
vec_strat = arrays(dtype=float, shape=2, elements=positives)
mat_strat = arrays(dtype=float, shape=(2, 2), elements=mvalues).map(minus_major)
time_strat = st.floats(min_value=0.5, max_value=3.0)


@settings(max_examples=100)
@given(m=mat_strat, v=vec_strat, t=time_strat)
def test_cram48_vs_pade_on_2x2(cram48_serial: Exponentiator, m: csr_matrix, v: np.ndarray, t: float):
    z = csr_matrix(np.zeros((2, 2)))
    cramsol = cram48_serial(m, z, lambda x: 0.0, v.copy(), t)
    padesol = spla.expm_multiply(m, v, 0.0, t, num=5, endpoint=True)[-1]
    assert np.allclose(cramsol, padesol, rtol=1e-3)


@settings(max_examples=200)
@given(d=mat_strat, r=mat_strat, n=vec_strat, t=time_strat)
def test_cram_is_same_as_magnus_for_constant_flux(
    cram48_serial: Exponentiator, magnus48: Exponentiator, d: csr_matrix, r: csr_matrix, n: np.array, t: float
):
    d, r = map(csr_matrix, (d, r))
    cramsol = cram48_serial(d, r, lambda x: 1.0, n.copy(), t)
    magnussol = magnus48(d, r, lambda x: 1.0, n, t)
    assert np.allclose(cramsol, magnussol)
