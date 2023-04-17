from .context import starktools
import numpy as np
from pytest import approx, raises


def test_matrixHs_comparison():
    h = starktools.MatrixHs(3,5)
    qd = starktools.QuantumDefects()
    assert (h[3][1][3][0] == qd.calc_matrix_element(3, 1, 3, 0, nmax = 5))
    assert (h[3][0][3][1] ==  qd.calc_matrix_element(3, 0, 3, 1, nmax = 5))
    assert (h[3][0][3][1] == h[3][1][3][0])