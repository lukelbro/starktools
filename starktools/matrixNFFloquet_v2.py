import functools
import numpy as np
from math import sqrt
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Tuple
from numpy.typing import NDArray

from .basis import nlqqbasis
from .tools import Tools

DEFECTS = {
    0: [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
    1: [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
    2: [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
    3: [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
    4: [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
    5: [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
    6: [2.3047609e-05, -0.0002610672, -4.04e-06, 0],
}

_nl_cached_functions = []


def _nl_cached(func):
    cached_func = functools.cache(func)
    _nl_cached_functions.append(cached_func)
    return cached_func


class quantumdefect_nl:

    @staticmethod
    def clear_cache() -> None:
        for func in _nl_cached_functions:
            func.cache_clear()

    @staticmethod
    def calc_defect(n: int, l: int) -> float:
        coeffs = DEFECTS.get(l, 0)
        if not isinstance(coeffs, (list, tuple)):
            return float(coeffs)
        defect = 0.0
        order = 0
        for d in coeffs:
            defect += d / (n - defect) ** order
            order += 2
        return defect

    @staticmethod
    def energy_level(n: int, l: int) -> float:
        ns = n - quantumdefect_nl.calc_defect(n, l)
        return -0.5 * ns ** -2

    @staticmethod
    @_nl_cached
    def numerov(ns: float, l: int, nmax: int) -> np.ndarray:
        return Tools.numerov(ns, l, nmax)

    @staticmethod
    @_nl_cached
    def radial_integral(n1: int, l1: int, n2: int, l2: int, nmax: int) -> float:
        ns1 = n1 - quantumdefect_nl.calc_defect(n1, l1)
        ns2 = n2 - quantumdefect_nl.calc_defect(n2, l2)
        wf1 = quantumdefect_nl.numerov(ns1, l1, nmax)
        wf2 = quantumdefect_nl.numerov(ns2, l2, nmax)
        return Tools.numerov_calc_matrix_element(wf1, wf2)

    @staticmethod
    def angular_factor(l1: int, l2: int, ml: int = 0) -> Tuple[float, float]:
        if l2 != l1 + 1:
            raise ValueError("angular_factor expects l2 = l1 + 1")
        A1 = sqrt((l2 ** 2 - ml ** 2) / ((2 * l2 + 1) * (2 * l2 - 1)))
        A2 = sqrt(((l1 + 1) ** 2 - ml ** 2) / ((2 * l1 + 3) * (2 * l1 + 1)))
        return A1, A2

    @staticmethod
    def matrix_element(
        n1: int, l1: int, n2: int, l2: int, nmax: int, ml: int = 0
    ) -> Tuple[float, float]:
        if abs(l1 - l2) != 1:
            return 0.0, 0.0
        l_low, l_high = min(l1, l2), max(l1, l2)
        R = quantumdefect_nl.radial_integral(n1, l1, n2, l2, nmax)
        A1, A2 = quantumdefect_nl.angular_factor(l_low, l_high, ml)
        if l1 < l2:
            return R * A1, R * A2
        else:
            return R * A2, R * A1


State = Tuple[int, ...]


class MatrixNFBase(ABC):
    basis: Callable[..., Iterable[State]]
    states: List[State]
    lookuptable: Dict[State, int]
    matrix: NDArray[np.float64]

    @abstractmethod
    def generate_matrix(self) -> NDArray[np.float64]:
        pass

    def generate_basis_states(self, **kwargs) -> Tuple[List[State], Dict[State, int]]:
        states: List[State] = []
        lookuptable: Dict[State, int] = {}
        for index, state in enumerate(self.basis(**kwargs)):
            states.append(state)
            lookuptable[state] = index
        return states, lookuptable

    def __array__(self, dtype=None) -> NDArray[np.float64]:
        return np.array(self.matrix, dtype=dtype)

    def __repr__(self) -> str:
        return str(self.matrix)

    def __getitem__(self, index: int):
        return self.matrix[index]


class MatrixH0NFFloquet(MatrixNFBase):
    def __init__(
        self,
        nmin: int,
        nmax: int,
        qmaxs: List[int],
        frequencies: List[float],
    ) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs
        self.frequencies = frequencies
        self.basis = nlqqbasis
        self.states, self.lookuptable = self.generate_basis_states(
            nmin=nmin, nmax=nmax, qmaxs=qmaxs
        )
        self.num_states = len(self.states)
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        matrix = np.zeros((self.num_states, self.num_states))
        for i, state in enumerate(self.states):
            n, l = state[0], state[1]
            q_vec = state[2:]
            E0 = quantumdefect_nl.energy_level(n, l)
            E_floquet = sum(self.frequencies[k] * q_vec[k] for k in range(len(q_vec)))
            matrix[i, i] = E0 + E_floquet
        return matrix


class MatrixHsNFFloquet(MatrixNFBase):
    def __init__(
        self,
        nmin: int,
        nmax: int,
        qmaxs: List[int],
    ) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs
        self.basis = nlqqbasis
        self.states, self.lookuptable = self.generate_basis_states(
            nmin=nmin, nmax=nmax, qmaxs=qmaxs
        )
        self.num_states = len(self.states)
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        matrix = np.zeros((self.num_states, self.num_states))

        lq_to_indices: Dict[Tuple, List[int]] = {}
        for idx, state in enumerate(self.states):
            key = (state[1], state[2:])
            lq_to_indices.setdefault(key, []).append(idx)

        for (l1, q1), indices1 in lq_to_indices.items():
            l2 = l1 + 1
            key2 = (l2, q1)
            if key2 not in lq_to_indices:
                continue
            for i in indices1:
                n1 = self.states[i][0]
                for j in lq_to_indices[key2]:
                    n2 = self.states[j][0]
                    v12, v21 = quantumdefect_nl.matrix_element(n1, l1, n2, l2, self.nmax)
                    matrix[i, j] = v12
                    matrix[j, i] = v21
        return matrix


class MatrixHfNFFloquet(MatrixNFBase):
    def __init__(
        self,
        nmin: int,
        nmax: int,
        qmaxs: List[int],
    ) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs
        self.N = len(qmaxs)
        self.basis = nlqqbasis
        self.states, self.lookuptable = self.generate_basis_states(
            nmin=nmin, nmax=nmax, qmaxs=qmaxs
        )
        self.num_states = len(self.states)
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> Tuple[NDArray[np.float64], ...]:
        N = self.N
        matrices: List[NDArray[np.float64]] = [
            np.zeros((self.num_states, self.num_states)) for _ in range(N)
        ]

        lq_to_indices: Dict[Tuple, List[int]] = {}
        for idx, state in enumerate(self.states):
            key = (state[1], state[2:])
            lq_to_indices.setdefault(key, []).append(idx)

        for (l1, q1), indices1 in lq_to_indices.items():
            l2 = l1 + 1
            for field_idx in range(N):
                for delta_q in (+1, -1):
                    q2 = list(q1)
                    q2[field_idx] += delta_q
                    key2 = (l2, tuple(q2))
                    if key2 not in lq_to_indices:
                        continue
                    for i in indices1:
                        n1 = self.states[i][0]
                        for j in lq_to_indices[key2]:
                            n2 = self.states[j][0]
                            v12, v21 = quantumdefect_nl.matrix_element(
                                n1, l1, n2, l2, self.nmax
                            )
                            matrices[field_idx][i, j] = 0.5 * v12
                            matrices[field_idx][j, i] = 0.5 * v21

        return tuple(matrices)

    def __call__(self, famps: List[float]) -> NDArray[np.float64]:
        if len(famps) != self.N:
            raise ValueError(f"Expected {self.N} amplitudes, got {len(famps)}")
        return sum(f * M for f, M in zip(famps, self.matrix))

    def __array__(self, dtype=None):
        raise TypeError(
            "MatrixHfNFFloquet stores per-field unit matrices. "
            "Use hf(famps) to get the assembled matrix."
        )
