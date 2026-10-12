import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray

from .basis import nlqqbasis
from .matrix_base import matrix
from .matrix_stark import angular_factor, radial_integral


def _lq_groups(states) -> Dict[Tuple, List[int]]:
    groups: Dict[Tuple, List[int]] = {}
    for idx, state in enumerate(states):
        groups.setdefault((state[1], state[2:]), []).append(idx)
    return groups


class MatrixH0NFFloquet(matrix):
    def __init__(
        self,
        nmin: int,
        nmax: int,
        qmaxs: List[int],
        frequencies: List[float],
        defects: Optional[Dict] = None,
    ) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs
        self.frequencies = frequencies
        self.defects = {} if defects is None else defects
        self.basis = nlqqbasis
        self.states, self.lookuptable = self.generate_basis_states(
            nmin=nmin, nmax=nmax, qmaxs=qmaxs
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        matrix = np.zeros((self.num_states, self.num_states))
        for i, state in enumerate(self.states):
            n, l = state[0], state[1]
            q_vec = state[2:]
            E0 = self.energy_level(n, l)
            E_floquet = sum(self.frequencies[k] * q_vec[k] for k in range(len(q_vec)))
            matrix[i, i] = E0 + E_floquet
        return matrix


class MatrixHsNFFloquet(matrix):
    def __init__(
        self,
        nmin: int,
        nmax: int,
        qmaxs: List[int],
        defects: Optional[Dict] = None,
    ) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs
        self.defects = {} if defects is None else defects
        self.basis = nlqqbasis
        self.states, self.lookuptable = self.generate_basis_states(
            nmin=nmin, nmax=nmax, qmaxs=qmaxs
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        matrix = np.zeros((self.num_states, self.num_states))
        ns = [n - self.quantum_defect(n, l) for n, l, *_ in self.states]
        groups = _lq_groups(self.states)

        for (l1, q1), indices1 in groups.items():
            l2 = l1 + 1
            for i in indices1:
                for j in groups.get((l2, q1), ()):
                    value = radial_integral(ns[i], l1, ns[j], l2, self.nmax) * angular_factor(l2)
                    matrix[i, j] = value
                    matrix[j, i] = value
        return matrix


class MatrixHfNFFloquet(matrix):
    def __init__(
        self,
        nmin: int,
        nmax: int,
        qmaxs: List[int],
        defects: Optional[Dict] = None,
    ) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs
        self.N = len(qmaxs)
        self.defects = {} if defects is None else defects
        self.basis = nlqqbasis
        self.states, self.lookuptable = self.generate_basis_states(
            nmin=nmin, nmax=nmax, qmaxs=qmaxs
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> Tuple[NDArray[np.float64], ...]:
        N = self.N
        matrices: List[NDArray[np.float64]] = [
            np.zeros((self.num_states, self.num_states)) for _ in range(N)
        ]
        ns = [n - self.quantum_defect(n, l) for n, l, *_ in self.states]
        groups = _lq_groups(self.states)

        for (l1, q1), indices1 in groups.items():
            l2 = l1 + 1
            for field_idx in range(N):
                for delta_q in (+1, -1):
                    q2 = list(q1)
                    q2[field_idx] += delta_q
                    for i in indices1:
                        for j in groups.get((l2, tuple(q2)), ()):
                            value = 0.5 * radial_integral(ns[i], l1, ns[j], l2, self.nmax) * angular_factor(l2)
                            matrices[field_idx][i, j] = value
                            matrices[field_idx][j, i] = value

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
