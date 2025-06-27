from sympy.physics.wigner import wigner_3j, wigner_6j
import numpy as np
from .basis import nlbasis, nlqbasis, nlqqbasis, qqbasis
from .tools import Tools
from .constants import Constants
import functools
from typing import Callable, Iterator, Tuple, Any

DEFECTS = {
    0: {  # S states (l=0)
        0: {  # singlet
            0: [0.1397180648621, 0.02783573718, 0.0167922941, -0.001459031, 0.002922765]
        },
        1: {  # triplet
            1: [0.2966564877175, 0.03829666659, 0.007513112, -0.004547679, 0.00218014]
        }
    },
    1: {  # P states (l=1)
        0: {
            1: [-0.01214180360364, 0.007519080459, 0.0139778015, 0.004837312, 0.001228329]
        },
        1: {
            0: [0.0683280025127, -0.01864197524, -0.0123316557, -0.007951545, -0.00544810],
            1: [0.0683578576527, -0.01863046224, -0.0123304057, -0.007951245, -0.00545010],
            2: [0.0683602837923, -0.01862922821, -0.0123327551, -0.007952741, -0.0054519]
        }
    },
    2: {  # D states (l=2)
        0: {
            2: [0.00211337846449, -0.003090051058, 0.0000082722, -0.000309431, -0.00040114]
        },
        1: {
            1: [0.00288558028122, -0.006357601227, 0.0003366711, 0.000839416, 0.000379872],
            2: [0.00289094149325, -0.006357183630, 0.0003377711, 0.000839216, 0.000432375],
            3: [0.00289132882526, -0.006357704033, 0.0003367013, 0.000839518, 0.000381183]
        }
    },
    3: {  # F states (l=3)
        0: {
            3: [0.0004402942662, -0.00168944665, -0.000118320, 0.00032618]
        },
        1: {
            2: [0.0004448698922, -0.00173927524, 0.0001047676, 0.000033769],
            3: [0.0004485948328, -0.00172723230, 0.00015249, -0.000248683],
            4: [0.0004473792721, -0.00173921723, 0.0001047871, 0.000033164]
        }
    },
    4: {  # G states (l=4)
        0: {
            4: [0.00012473449079, -0.00079623012, -0.0001020553, -0.000013669]
        },
        1: {
            3: [0.0001257074312, -0.00079649819, -0.0000098081, -0.00001911],
            4: [0.0001287131610, -0.00079624615, -0.0000118966, -0.000014185],
            5: [0.0001271416711, -0.00079648417, -0.0000098575, -0.00001910]
        }
    },
    5: {  # H states (l=5)
        0: {
            5: [0.00004710089961, -0.000433227784, -0.0000081426]
        },
        1: {
            4: [0.00004779706743, -0.000433232255, -0.0000080716],
            5: [0.00004975761451, -0.000433227465, -0.0000081319],
            6: [0.00004872984645, -0.000433228157, -0.0000081016]
        }
    },
    6: {  # I states (l=6)
        0: {
            6: [0.00002186888117, -0.000261067322, -0.00000404867]
        },
        1: {
            5: [0.00002239075920, -0.000261068028, -0.00000404287],
            6: [0.00002376848314, -0.000261066218, -0.00000407658],
            7: [0.00002304760926, -0.000261067235, -0.0000040411]
        }
    }
}

_cached_functions = []

def cached(func):
    cached_func = functools.cache(func)
    _cached_functions.append(cached_func)
    return cached_func
    
class quantumdefect_fs:
    @staticmethod
    def clear_all_cache():
        for func in _cached_functions:
            func.cache_clear()

    def calc_defect(n, l, s, j):
        '''where $n^* = n - \delta(n^*)$, and the quantum defect $\delta(n^*)$ is defined recursively through the Ritz expansion
            \begin{equation}
            \delta(n^*) = \delta_0 + \frac{\delta_2}{(n - \delta)^2} + \frac{\delta_4}{(n - \delta)^4} + \cdots
            \end{equation}'''
        try:
            deltas = DEFECTS[l][s][j]
        except:
            return 0
        defect = 0
        order = 0
        for d in deltas:
            defect += d/((n-defect)**order)
            order += 2
        return defect
    
    def energylevel( n, l, s, j):
        """
        Calculate the energy level of an electron in an atom based on quantum numbers and defects.

        Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.
        s (int): Spin quantum number.
        j (float): Total angular momentum quantum number.

        Returns:
        float: The energy level of the electron in atomic units.

        The energy level is calculated using the formula:
        E = - (n - self.calc_defects(n, l, s, j))**(-2)
        where calc_defects is a method that computes the quantum defects.
        """
        return - (n - quantumdefect_fs.calc_defect(n,l,s,j))**(-2)
    
    @staticmethod
    @cached
    def radial_integral(n1, l1, s1, j1, n2, l2, s2, j2, nmax):
        """
        Calculate the radial integral of two electron wavefunctions considering quantum defects.

        Parameters:
        n1 (int): Principal quantum number of the first energy level.
        l1 (int): Orbital angular momentum quantum number of the first energy level.
        s1 (int): Spin quantum number of the first energy level.
        j1 (float): Total angular momentum quantum number of the first energy level.
        n2 (int): Principal quantum number of the second energy level.
        l2 (int): Orbital angular momentum quantum number of the second energy level.
        s2 (int): Spin quantum number of the second energy level.
        j2 (float): Total angular momentum quantum number of the second energy level.
        nmax (int): Maximum quantum number used for the Numerov integration.

        Returns:
        float: The radial integral value representing the overlap of the two wavefunctions.

        The method performs the following steps:
        1. Computes the effective principal quantum numbers (ns1, ns2) by subtracting the quantum defects from n1 and n2.
        2. Uses the Numerov method to compute the wavefunctions (wf1, wf2).
        3. Calculates the radial overlap integral using the computed wavefunctions.

        Notes:
        - The `calc_defect` method is used to determine the quantum defects.
        - The `st.Tools.numerov` function is employed to generate the wavefunctions.
        - The `st.Tools.numerov_calc_matrix_element` function calculates the matrix element for the overlap integral.
        """   
        # Account for Quantum defects
        ns1 = n1 - quantumdefect_fs.calc_defect(n1, l1, s1, j1)
        ns2 = n2 - quantumdefect_fs.calc_defect(n2, l2, s2, j2)

        # Numerov integrals
        wf1 = quantumdefect_fs.numerov(ns1, l1, nmax)
        wf2 = quantumdefect_fs.numerov(ns2, l2, nmax)
        
        # Radial overlap
        radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)
        return radialInt
    
    @staticmethod
    @cached
    def numerov(n, l, nmax):
            return Tools.numerov(n, l, nmax)

    @staticmethod
    @cached
    def angular_integral(n1, l1, j1, mj1, n2, l2, j2, mj2, q, s, nmax):
        max_l = np.max([l1, l2])
        phase = (-1)**(-mj2+s+max_l)

        A = np.sqrt(max_l)*np.sqrt((2*j1+1)*(2*j2+1)) 
        W6 = wigner_6j(l2, j2, s, j1, l1, 1)
        W3 = wigner_3j(j1, 1, j2, mj1, q, -mj2)

        return float(phase * A * W6 * W3)
    
    @staticmethod
    @cached
    def dipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, q, s, nmax):
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
        n1 (int): Principal quantum number of the initial state.
        l1 (int): Orbital angular momentum quantum number of the initial state.
        j1 (float): Total angular momentum quantum number of the initial state.
        mj1 (int): Magnetic quantum number of the initial state.
        n2 (int): Principal quantum number of the final state.
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
        2. Computes the radial integral using the `radial_integral` method, which accounts for quantum defects and wavefunction overlap.
        3. Computes the phase factor using the orbital and magnetic quantum numbers.
        4. Calculates the prefactor `A` based on the total angular momentum quantum numbers.
        5. Uses Wigner 6-j and 3-j symbols to compute the angular integral components.
        6. Combines the radial and angular integrals to obtain the dipole matrix element.
        """

        if np.abs(l1-l2) != 1:
            return 0
    
        radialInt = quantumdefect_fs.radial_integral(n1, l1, s, j1, n2, l2, s, j2, nmax)
        angularInt = quantumdefect_fs.angular_integral(n1, l1, j1, mj1, n2, l2, j2, mj2, q, s, nmax)
        return float(angularInt * radialInt)


    def calc_transition( n1, l1, s1, j1, n2, l2, s2, j2):
        """
        Calculate the transition frequency between two atomic energy levels in GHz.

        Parameters:
        n1 (int): Principal quantum number of the initial state.
        l1 (int): Orbital angular momentum quantum number of the initial state.
        s1 (float): Spin quantum number of the initial state.
        j1 (float): Total angular momentum quantum number of the initial state.
        n2 (int): Principal quantum number of the final state.
        l2 (int): Orbital angular momentum quantum number of the final state.
        s2 (float): Spin quantum number of the final state.
        j2 (float): Total angular momentum quantum number of the final state.

        Returns:
        float: The transition frequency in gigahertz (GHz).
        """
        return Constants.c/(1/((quantumdefect_fs.energylevel(n1, l1, s1, j1)*Constants.R_He  - quantumdefect_fs.energylevel(n2, l2, s2, j2)*Constants.R_He)))*10**-9

class basis_fs:
    @staticmethod
    def float_range(start, stop, step):
        current = start
        while current < stop:
            yield current
            current += step

    @staticmethod
    def generate_atom_levels_fs(n_min, n_max, S, lmax=None):
        """
        Generate quantum numbers for atomic levels in fine structure.

        This is an iterator method that yields tuples representing the quantum numbers (n, l, S, j, m_j) for atomic levels
        within the specified range.

        Parameters:
        n_min (int): The minimum principal quantum number.
        n_max (int): The maximum principal quantum number.
        S (float): The spin quantum number.
        lmax (int, optional): The maximum orbital angular momentum quantum number. Defaults to None.

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
                            yield n, l, S, j, m_j
    
    @staticmethod
    def generate_atom_levels_fs_floquet(n_min, n_max, S, qmax, lmax=None):
        for n, l, S, j, m_j in basis_fs.generate_atom_levels_fs(n_min, n_max, S, lmax=None):
            q = - qmax
            while q <= qmax:
                yield (n, l, S, j, m_j, q)
                q += 1

    


class MatrixFs:
    def __init__(self, nmin: int, nmax: int, S: float, polarization: int, basis=None, frequency: float = None):
        self.nmin = nmin
        self.nmax = nmax
        self.S = S
        self.polarization = polarization
        self.frequency = frequency
        if basis is None:
            self.basis = basis_fs.generate_atom_levels_fs(nmin, nmax, S)
        else:
            self.basis = basis
        self.states, self.lookuptable = self.generate_matrix_states()
        self.num_states = len(self.states)

        self.matrix = self.generate_matrix()

    def generate_matrix_states(self):
        states = []
        lookuptable = {}
        index = 0
        for state in self.basis:
            states.append(state)
            lookuptable[state] = index
            index += 1
        return states, lookuptable

    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        return matrix
    
    def convert_state_to_index(self, state):
        return self.lookuptable[state]

    def convert_states_to_index(self, state1, state2):
        return self.lookuptable[state1], self.lookuptable[state2]

    def convert_index_to_state(self, index):
        return self.states[index]

    def convert_indexs_to_states(self, index1, index2):
        return self.states[index1], self.states[index2]
        
    def __repr__(self):
        return str(self.matrix)
    
    def __getitem__(self, index):
        return self.matrix[index]

class MatrixFsH0(MatrixFs):
    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        for i, state in enumerate(self.states):
            n, l, s, j, m = state
            matrix[i,i] = 0.5* quantumdefect_fs.energylevel(n, l, s, j)
        return matrix

class MatrixFsHs(MatrixFs):
    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))

        # Make a dictionary of states with the l value as the keys
        l_to_indices = {}
        for idx, state in enumerate(self.states):
            l = state[1]
            l_to_indices.setdefault(l, []).append(idx)

        # Fill matrix based on requirement for l pm 1.
        for l1, indices1 in l_to_indices.items():
            for delta in [-1, 1]:
                l2 = l1 + delta
                if l2 in l_to_indices:
                    for i in indices1:
                        n1, l1, s, j1, mj1 = self.states[i]
                        for j in l_to_indices[l2]:
                            n2, l2, s, j2, mj2 = self.states[j]
                            matrix[i, j] = quantumdefect_fs.dipoleMatrixElement(
                                n1, l1, j1, mj1, n2, l2, j2, mj2, self.polarization, s, self.nmax
                            )
        return matrix
    
class MatrixFsFloquet(MatrixFs):
    def __init__(self, nmin: int, nmax: int, S: float, polarization: int, qmax: int, frequency: float):
        self.qmax = qmax
        self.frequency = frequency
        floquet_basis = basis_fs.generate_atom_levels_fs_floquet(nmin, nmax, S, qmax)
        super().__init__(nmin, nmax, S, polarization, basis=floquet_basis, frequency=frequency)

class MatrixFsH0Floquet(MatrixFsFloquet):
    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        for i, state in enumerate(self.states):
            n, l, s, j, m, q = state
            matrix[i,i] = 0.5 *quantumdefect_fs.energylevel(n, l, s, j) + self.frequency * q
        return matrix
    
class MatrixFsHsFloquet(MatrixFsFloquet):    
    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        # Group indices by (l, q)
        lq_to_indices = {}
        for idx, state in enumerate(self.states):
            l = state[1]
            q = state[-1]
            lq_to_indices.setdefault((l, q), []).append(idx)

        # Fill matrix: only connect states with same q and l2 = l1 ± 1
        for (l1, q), indices1 in lq_to_indices.items():
            for delta in [-1, 1]:
                l2 = l1 + delta
                key2 = (l2, q)
                if key2 in lq_to_indices:
                    for i in indices1:
                        n1, l1, s, j1, mj1, q1 = self.states[i]
                        for j in lq_to_indices[key2]:
                            n2, l2, s, j2, mj2, q2 = self.states[j]
                            # q1 == q2 is always true here by construction
                            matrix[i, j] = quantumdefect_fs.dipoleMatrixElement(
                                n1, l1, j1, mj1, n2, l2, j2, mj2, self.polarization, s, self.nmax
                            )
        return matrix

class MatrixFsHfFloquet(MatrixFsFloquet):
    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        # Group indices by (l, q)
        lq_to_indices = {}
        for idx, state in enumerate(self.states):
            l = state[1]
            q = state[-1]
            lq_to_indices.setdefault((l, q), []).append(idx)

        # Fill matrix: connect states with delta l = ±1 and delta q = ±1
        for (l1, q1), indices1 in lq_to_indices.items():
            for delta_l in [-1, 1]:
                l2 = l1 + delta_l
                for delta_q in [-1, 1]:
                    q2 = q1 + delta_q
                    key2 = (l2, q2)
                    if key2 in lq_to_indices:
                        for i in indices1:
                            n1, l1, s, j1, mj1, q1 = self.states[i]
                            for j in lq_to_indices[key2]:
                                n2, l2, s, j2, mj2, q2 = self.states[j]
                                matrix[i, j] = 0.5 * quantumdefect_fs.dipoleMatrixElement(
                                    n1, l1, j1, mj1, n2, l2, j2, mj2, self.polarization, s, self.nmax
                                )
        return matrix