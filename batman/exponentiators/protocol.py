"""Exponentiator agreed upon protocol

"""
from typing import Protocol, Callable

from scipy.sparse import csr_matrix
from numpy import ndarray

from batman.units import Second

FluxFunc = Callable[[Second], float]


class Exponentiator(Protocol):
    """A Protocol for exponentiators. Basically a function object with an agreed
    upon signature.

    """

    def __call__(self, d: csr_matrix, r: csr_matrix, flux: FluxFunc,
                 n0: ndarray, dt: Second) -> ndarray:
        raise NotImplementedError
