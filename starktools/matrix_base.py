from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .defects import lookup_defect

State = Tuple[Any, ...]


class matrix(ABC):
    """Base class for Hamiltonian matrices built on a quantum-number basis.

    Subclasses set ``self.basis`` (a basis callable), build ``self.states`` and
    ``self.lookuptable`` via :meth:`generate_basis_states`, and implement
    :meth:`generate_matrix`.
    """

    basis: Callable[..., Iterable[State]]
    states: List[State]
    lookuptable: Dict[State, int]
    matrix: NDArray[np.float64]
    defects: Dict = {}

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

    def quantum_defect(self, n: int, l: int, s: Optional[float] = None, j: Optional[float] = None) -> float:
        return lookup_defect(self.defects, n, l, s, j)

    def energy_level(self, n: int, l: int, s: Optional[float] = None, j: Optional[float] = None) -> float:
        return -0.5 * (n - self.quantum_defect(n, l, s, j)) ** -2

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.matrix.shape

    def convert_state_to_index(self, state: State) -> int:
        return self.lookuptable[state]

    def convert_states_to_index(self, state1: State, state2: State) -> Tuple[int, int]:
        return self.lookuptable[state1], self.lookuptable[state2]

    def convert_index_to_state(self, index: int) -> State:
        return self.states[index]

    def convert_indices_to_states(self, index1: int, index2: int) -> Tuple[State, State]:
        return self.states[index1], self.states[index2]

    def __array__(self, dtype=None) -> NDArray[np.float64]:
        return np.asarray(self.matrix, dtype=dtype)

    def __repr__(self) -> str:
        return str(self.matrix)

    def __getitem__(self, index: int):
        return self.matrix[index]

    def _check_compatible(self, other: "_MatrixType") -> None:
        if self.states != other.states:
            raise ValueError("matrices are built on different basis states")

    def __add__(self, other: "_MatrixType") -> "_matrix_view":
        self._check_compatible(other)
        return self._from_array(np.asarray(self.matrix) + np.asarray(other.matrix))

    def __sub__(self, other: "_MatrixType") -> "_matrix_view":
        self._check_compatible(other)
        return self._from_array(np.asarray(self.matrix) - np.asarray(other.matrix))

    def __mul__(self, scalar: float) -> "_matrix_view":
        return self._from_array(np.asarray(self.matrix) * scalar)

    def __rmul__(self, scalar: float) -> "_matrix_view":
        return self._from_array(scalar * np.asarray(self.matrix))

    def _from_array(self, array: NDArray[np.float64]) -> "_matrix_view":
        """Wrap a numpy array in a matrix view sharing this basis."""
        new = _matrix_view.__new__(_matrix_view)
        new.basis = self.basis
        new.states = self.states
        new.lookuptable = self.lookuptable
        new.defects = self.defects
        new.matrix = array
        return new


_MatrixType = matrix


class _matrix_view(matrix):
    """A matrix holding a computed array on an existing basis."""

    def generate_matrix(self) -> NDArray[np.float64]:
        return self.matrix
