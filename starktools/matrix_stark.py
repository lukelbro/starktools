import functools
from math import sqrt
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .basis import nlbasis
from .matrix_base import State, matrix
from .tools import Tools


@functools.cache
def _numerov(ns: float, l: int, nmax: int) -> Tools.numerov:
    return Tools.numerov(ns, l, nmax)


@functools.cache
def radial_integral(ns1: float, l1: int, ns2: float, l2: int, nmax: int) -> float:
    """Radial overlap of Numerov wavefunctions at effective quantum numbers ns1, ns2."""
    return Tools.numerov_calc_matrix_element(_numerov(ns1, l1, nmax), _numerov(ns2, l2, nmax))


def angular_factor(l2: int, ml: int = 0) -> float:
    """Angular part of <l2 - 1, ml| z |l2, ml>."""
    return sqrt((l2 ** 2 - ml ** 2) / ((2 * l2 + 1) * (2 * l2 - 1)))


def clear_cache() -> None:
    _numerov.cache_clear()
    radial_integral.cache_clear()


class matrixH0(matrix):
    """Field-free (diagonal) Hamiltonian on the (n, l) basis."""

    def __init__(self, nmin: int, nmax: int, ml: int = 0, defects: Optional[Dict] = None) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.ml = ml
        self.defects = {} if defects is None else defects
        self.basis = nlbasis
        self.states, self.lookuptable = self.generate_basis_states(nmin=nmin, nmax=nmax, ml=ml)
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        m = np.zeros((self.num_states, self.num_states))
        for i, (n, l) in enumerate(self.states):
            m[i, i] = self.energy_level(n, l)
        return m


class matrixHs(matrix):
    """DC Stark coupling Hamiltonian on the (n, l) basis."""

    def __init__(self, nmin: int, nmax: int, ml: int = 0, defects: Optional[Dict] = None) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.ml = ml
        self.defects = {} if defects is None else defects
        self.basis = nlbasis
        self.states, self.lookuptable = self.generate_basis_states(nmin=nmin, nmax=nmax, ml=ml)
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        m = np.zeros((self.num_states, self.num_states))
        for i, (n1, l1) in enumerate(self.states):
            ns1 = n1 - self.quantum_defect(n1, l1)
            for j, (n2, l2) in enumerate(self.states):
                if l2 != l1 + 1:
                    continue
                ns2 = n2 - self.quantum_defect(n2, l2)
                value = radial_integral(ns1, l1, ns2, l2, self.nmax) * angular_factor(l2, self.ml)
                m[i, j] = value
                m[j, i] = value
        return m
