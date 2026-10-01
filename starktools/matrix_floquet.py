from typing import Dict, List, Optional

from .matrixNFFloquet_v2 import (
    MatrixH0NFFloquet,
    MatrixHsNFFloquet,
    MatrixHfNFFloquet,
)


class matrixH0F(MatrixH0NFFloquet):
    """Single-field Floquet diagonal Hamiltonian (one oscillating field)."""

    def __init__(self, nmin: int, nmax: int, q: int, frequency: float) -> None:
        super().__init__(nmin, nmax, [q], [frequency])


class matrixHsF(MatrixHsNFFloquet):
    """Single-field Floquet DC coupling Hamiltonian."""

    def __init__(self, nmin: int, nmax: int, q: int) -> None:
        super().__init__(nmin, nmax, [q])


class matrixHfF(MatrixHfNFFloquet):
    """Single-field Floquet AC coupling Hamiltonian.

    Stores the per-field unit matrix (with the 1/2 sideband factor); assemble the
    full matrix for a given field amplitude by calling the instance: ``hf([f])``.
    """

    def __init__(self, nmin: int, nmax: int, q: int) -> None:
        super().__init__(nmin, nmax, [q])
