import functools
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sympy.physics.wigner import wigner_3j, wigner_6j

from .defects import HELIUM_FS
from .matrix_base import matrix
from .tools import Tools

_fs_cached_functions = []


def _fs_cached(func):
    cached_func = functools.cache(func)
    _fs_cached_functions.append(cached_func)
    return cached_func


class quantumdefect_fs:
    @staticmethod
    def clear_cache():
        for func in _fs_cached_functions:
            func.cache_clear()

    @staticmethod
    @_fs_cached
    def radial_integral(ns1, l1, ns2, l2, nmax):
        """Radial overlap of Numerov wavefunctions at effective quantum numbers ns1, ns2."""
        wf1 = quantumdefect_fs.numerov(ns1, l1, nmax)
        wf2 = quantumdefect_fs.numerov(ns2, l2, nmax)
        return Tools.numerov_calc_matrix_element(wf1, wf2)

    @staticmethod
    @_fs_cached
    def numerov(n, l, nmax):
            return Tools.numerov(n, l, nmax)

    @staticmethod
    @_fs_cached
    def angular_integral(l1, j1, mj1, l2, j2, mj2, q, s):
        max_l = np.max([l1, l2])
        phase = (-1)**(-mj2+s+max_l)

        A = np.sqrt(max_l)*np.sqrt((2*j1+1)*(2*j2+1)) 
        W6 = wigner_6j(l2, j2, s, j1, l1, 1)
        W3 = wigner_3j(j1, 1, j2, mj1, q, -mj2)

        return float(phase * A * W6 * W3)
    
    @staticmethod
    @_fs_cached
    def dipole_matrix_element(ns1, l1, j1, mj1, ns2, l2, j2, mj2, q, s, nmax):
        """
        Calculate the dipole matrix element for a transition between two fine-structure states in an atom. 
        This method takes into account the angular momentum of the photon.

        \begin{aligned}
        \mu_{eg} &= e(-1)^{L' + S - M'_J} \sqrt{(2J + 1)(2J' + 1)} \\
        &\quad \times \begin{Bmatrix}
        L' & J' & S \\
        J & L & 1
        \end{Bmatrix}
        \begin{pmatrix}
        J & 1 & J' \\
        M_J & q & -M'_J
        \end{pmatrix}
        \langle \alpha' L' \| r \| \alpha L \rangle.
        \end{aligned}

        Parameters:
        ns1 (float): Effective principal quantum number n - delta of the initial state.
        l1 (int): Orbital angular momentum quantum number of the initial state.
        j1 (float): Total angular momentum quantum number of the initial state.
        mj1 (int): Magnetic quantum number of the initial state.
        ns2 (float): Effective principal quantum number n - delta of the final state.
        l2 (int): Orbital angular momentum quantum number of the final state.
        j2 (float): Total angular momentum quantum number of the final state.
        mj2 (int): Magnetic quantum number of the final state.
        q (int): Projection of the angular momentum of the photon.
        s (int): Spin quantum number, assumed to be the same for both states.
        nmax (int): Maximum quantum number used for the Numerov integration.

        Returns:
        np.double: The dipole matrix element value.

        The method performs the following steps:
        1. Sets the spin quantum numbers (s1, s2) for both states to the provided spin value `s`.
        2. Computes the radial integral using the `radial_integral` method (Numerov wavefunction overlap).
        3. Computes the phase factor using the orbital and magnetic quantum numbers.
        4. Calculates the prefactor `A` based on the total angular momentum quantum numbers.
        5. Uses Wigner 6-j and 3-j symbols to compute the angular integral components.
        6. Combines the radial and angular integrals to obtain the dipole matrix element.
        """

        if np.abs(l1-l2) != 1:
            return 0
    
        radialInt = quantumdefect_fs.radial_integral(ns1, l1, ns2, l2, nmax)
        angularInt = quantumdefect_fs.angular_integral(l1, j1, mj1, l2, j2, mj2, q, s)
        return float(angularInt * radialInt)


class basis_fs:
    @staticmethod
    def float_range(start, stop, step):
        current = start
        while current < stop:
            yield current
            current += step

    @staticmethod
    def generate_atom_levels_fs(n_min, n_max, S, lmax=None, mjmax=None):
        """
        Generate quantum numbers for atomic levels in fine structure.

        This is an iterator method that yields tuples representing the quantum numbers (n, l, S, j, m_j) for atomic levels
        within the specified range.

        Parameters:
        n_min (int): The minimum principal quantum number.
        n_max (int): The maximum principal quantum number.
        S (float): The spin quantum number.
        lmax (int, optional): The maximum orbital angular momentum quantum number. Defaults to None.
        mjmax (float, optional): The maximum magnetic quantum number. Defaults to None.

        Yields:
        tuple: A tuple representing the quantum numbers (n, l, S, j, m_j) for each level.

        The method iterates over possible values of principal quantum number (n), orbital angular momentum quantum number (l),
        total angular momentum quantum number (j), and magnetic quantum number (m_j), yielding all valid combinations within
        the specified range.
        """
        for n in range(n_min, n_max + 1):
            for l in range(n):
                if lmax is None or l <= lmax:
                    for j in basis_fs.float_range(abs(l - S), l + S + 1, 1):
                        for m_j in basis_fs.float_range(-j, j + 1, 1):
                            if mjmax is None or abs(m_j) <= mjmax:
                                yield n, l, S, j, m_j
    
    @staticmethod
    def generate_atom_levels_fs_floquet(n_min, n_max, S, qmax, lmax=None, mjmax=None):
        for n, l, S, j, m_j in basis_fs.generate_atom_levels_fs(n_min, n_max, S, lmax=lmax, mjmax=mjmax):
            q = - qmax
            while q <= qmax:
                yield (n, l, S, j, m_j, q)
                q += 1


def _group_indices(states, key):
    groups = {}
    for idx, state in enumerate(states):
        groups.setdefault(key(state), []).append(idx)
    return groups


def _dipole(m, state1, state2):
    n1, l1, s, j1, mj1 = state1[:5]
    n2, l2, _, j2, mj2 = state2[:5]
    ns1 = n1 - m.quantum_defect(n1, l1, s, j1)
    ns2 = n2 - m.quantum_defect(n2, l2, s, j2)
    return quantumdefect_fs.dipole_matrix_element(ns1, l1, j1, mj1, ns2, l2, j2, mj2, m.polarization, s, m.nmax)


class matrixfsH0(matrix):
    """Field-free (diagonal) Hamiltonian on the (n, l, S, j, mj) basis."""

    def __init__(self, nmin: int, nmax: int, S: float, lmax: Optional[int] = None, mjmax: Optional[float] = None, defects: Optional[Dict] = None) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.S = S
        self.lmax = lmax
        self.mjmax = mjmax
        self.defects = HELIUM_FS if defects is None else defects
        self.basis = basis_fs.generate_atom_levels_fs
        self.states, self.lookuptable = self.generate_basis_states(
            n_min=nmin, n_max=nmax, S=S, lmax=lmax, mjmax=mjmax
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        m = np.zeros((self.num_states, self.num_states))
        for i, (n, l, s, j, mj) in enumerate(self.states):
            m[i, i] = self.energy_level(n, l, s, j)
        return m


class matrixfsHs(matrix):
    """DC Stark coupling Hamiltonian on the (n, l, S, j, mj) basis."""

    def __init__(self, nmin: int, nmax: int, S: float, polarization: int, lmax: Optional[int] = None, mjmax: Optional[float] = None, defects: Optional[Dict] = None) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.S = S
        self.polarization = polarization
        self.lmax = lmax
        self.mjmax = mjmax
        self.defects = HELIUM_FS if defects is None else defects
        self.basis = basis_fs.generate_atom_levels_fs
        self.states, self.lookuptable = self.generate_basis_states(
            n_min=nmin, n_max=nmax, S=S, lmax=lmax, mjmax=mjmax
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        m = np.zeros((self.num_states, self.num_states))
        groups = _group_indices(self.states, lambda s: s[1])
        for l1, indices1 in groups.items():
            for l2 in (l1 - 1, l1 + 1):
                for i in indices1:
                    for j in groups.get(l2, ()):
                        m[i, j] = _dipole(self, self.states[i], self.states[j])
        return m


class matrixfsH0F(matrix):
    """Single-field Floquet diagonal Hamiltonian on the (n, l, S, j, mj, q) basis."""

    def __init__(self, nmin: int, nmax: int, S: float, qmax: int, frequency: float, lmax: Optional[int] = None, mjmax: Optional[float] = None, defects: Optional[Dict] = None) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.S = S
        self.qmax = qmax
        self.frequency = frequency
        self.lmax = lmax
        self.mjmax = mjmax
        self.defects = HELIUM_FS if defects is None else defects
        self.basis = basis_fs.generate_atom_levels_fs_floquet
        self.states, self.lookuptable = self.generate_basis_states(
            n_min=nmin, n_max=nmax, S=S, qmax=qmax, lmax=lmax, mjmax=mjmax
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        m = np.zeros((self.num_states, self.num_states))
        for i, (n, l, s, j, mj, q) in enumerate(self.states):
            m[i, i] = self.energy_level(n, l, s, j) + self.frequency * q
        return m


class matrixfsHsF(matrix):
    """Single-field Floquet DC coupling Hamiltonian (couples equal q only)."""

    def __init__(self, nmin: int, nmax: int, S: float, polarization: int, qmax: int, lmax: Optional[int] = None, mjmax: Optional[float] = None, defects: Optional[Dict] = None) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.S = S
        self.polarization = polarization
        self.qmax = qmax
        self.lmax = lmax
        self.mjmax = mjmax
        self.defects = HELIUM_FS if defects is None else defects
        self.basis = basis_fs.generate_atom_levels_fs_floquet
        self.states, self.lookuptable = self.generate_basis_states(
            n_min=nmin, n_max=nmax, S=S, qmax=qmax, lmax=lmax, mjmax=mjmax
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> NDArray[np.float64]:
        m = np.zeros((self.num_states, self.num_states))
        groups = _group_indices(self.states, lambda s: (s[1], s[5]))
        for (l1, q), indices1 in groups.items():
            for l2 in (l1 - 1, l1 + 1):
                for i in indices1:
                    for j in groups.get((l2, q), ()):
                        m[i, j] = _dipole(self, self.states[i], self.states[j])
        return m


class matrixfsHfF(matrix):
    """Single-field Floquet AC coupling Hamiltonian (couples q to q +/- 1).

    Stores the unit matrix (with the 1/2 sideband factor); assemble the full
    matrix for a given field amplitude by calling the instance: ``hf([f])``.
    """

    def __init__(self, nmin: int, nmax: int, S: float, polarization: int, qmax: int, lmax: Optional[int] = None, mjmax: Optional[float] = None, defects: Optional[Dict] = None) -> None:
        self.nmin = nmin
        self.nmax = nmax
        self.S = S
        self.polarization = polarization
        self.qmax = qmax
        self.lmax = lmax
        self.mjmax = mjmax
        self.defects = HELIUM_FS if defects is None else defects
        self.basis = basis_fs.generate_atom_levels_fs_floquet
        self.states, self.lookuptable = self.generate_basis_states(
            n_min=nmin, n_max=nmax, S=S, qmax=qmax, lmax=lmax, mjmax=mjmax
        )
        self.matrix = self.generate_matrix()

    def generate_matrix(self) -> Tuple[NDArray[np.float64]]:
        m = np.zeros((self.num_states, self.num_states))
        groups = _group_indices(self.states, lambda s: (s[1], s[5]))
        for (l1, q1), indices1 in groups.items():
            for l2 in (l1 - 1, l1 + 1):
                for q2 in (q1 - 1, q1 + 1):
                    for i in indices1:
                        for j in groups.get((l2, q2), ()):
                            m[i, j] = 0.5 * _dipole(self, self.states[i], self.states[j])
        return (m,)

    def __call__(self, famps: List[float]) -> NDArray[np.float64]:
        if len(famps) != 1:
            raise ValueError(f"Expected 1 amplitude, got {len(famps)}")
        return famps[0] * self.matrix[0]

    def __array__(self, dtype=None):
        raise TypeError(
            "matrixfsHfF stores the unit matrix. "
            "Use hf(famps) to get the assembled matrix."
        )
