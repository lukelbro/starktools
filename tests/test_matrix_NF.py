import starktools
import numpy as np
import pytest
from pytest import approx
from starktools import (
    matrixH0NF,
    matrixHsNF,
    matrixHfNF,
)
from starktools import matrix_stark
from starktools.matrix_base import matrix
from starktools.basis import nlbasis

DEFECTS = {
    0: [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
    1: [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
    2: [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
    3: [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
    4: [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
    5: [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
    6: [2.3047609e-05, -0.0002610672, -4.04e-06, 0],
}
QD = starktools.QuantumDefects(DEFECTS)


# ---------------------------------------------------------------------------
# matrixH0NF
# ---------------------------------------------------------------------------

def test_h0_diagonal_structure():
    nmin, nmax = 2, 4
    qmaxs = [1, 1]
    freqs = [3.0, 5.0]
    h0 = matrixH0NF(nmin, nmax, qmaxs, freqs, defects=DEFECTS)

    for i, state in enumerate(h0.states):
        n, l = state[0], state[1]
        q_vec = state[2:]
        ns = n - QD.quantum_defect(n, l)
        expected = -0.5 * ns ** -2 + sum(freqs[k] * q_vec[k] for k in range(len(freqs)))
        assert h0.matrix[i, i] == approx(expected)

    off_diag = h0.matrix.copy()
    np.fill_diagonal(off_diag, 0)
    assert np.all(off_diag == 0)


# ---------------------------------------------------------------------------
# matrixHsNF
# ---------------------------------------------------------------------------

def test_hs_selection_rules():
    hs = matrixHsNF(3, 5, [1, 1], defects=DEFECTS)
    for i, s1 in enumerate(hs.states):
        n1, l1, *q1 = s1
        for j, s2 in enumerate(hs.states):
            n2, l2, *q2 = s2
            if q1 != q2 or abs(l1 - l2) != 1:
                assert hs.matrix[i, j] == 0, f"Nonzero at ({s1}, {s2})"


def test_hs_values_match_quantumdefects():
    nmin, nmax = 3, 5
    hs = matrixHsNF(nmin, nmax, [1, 1], defects=DEFECTS)
    qd = QD
    reference = qd.calc_matrix_element(3, 0, 3, 1, nmax=nmax)

    for q in range(-1, 2):
        for p in range(-1, 2):
            i = hs.lookuptable[(3, 0, q, p)]
            j = hs.lookuptable[(3, 1, q, p)]
            assert hs.matrix[i, j] == approx(reference, rel=1e-4)


def test_hs_hermitian():
    hs = matrixHsNF(3, 5, [1, 1], defects=DEFECTS)
    assert hs.matrix == approx(hs.matrix.T)


# ---------------------------------------------------------------------------
# matrixHfNF
# ---------------------------------------------------------------------------

def test_hf_num_unit_matrices():
    hf2 = matrixHfNF(3, 5, [1, 1], defects=DEFECTS)
    assert len(hf2.matrix) == 2

    hf3 = matrixHfNF(3, 5, [1, 1, 1], defects=DEFECTS)
    assert len(hf3.matrix) == 3


def test_hf_unit_matrices_disjoint():
    hf = matrixHfNF(3, 5, [2, 2], defects=DEFECTS)
    masks = [M != 0 for M in hf.matrix]
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            assert not np.any(masks[a] & masks[b]), \
                f"Matrices {a} and {b} share nonzero elements"


def test_hf_call_linearity():
    hf = matrixHfNF(3, 5, [1, 1], defects=DEFECTS)
    k = 3.7
    diff = hf([k, 1.0]) - hf([1.0, 1.0])
    expected = (k - 1) * hf.matrix[0]
    assert diff == approx(expected)


def test_hf_selection_rules():
    hf = matrixHfNF(3, 5, [1, 1], defects=DEFECTS)
    M = sum(hf.matrix)
    for i, s1 in enumerate(hf.states):
        n1, l1, *q1 = s1
        for j, s2 in enumerate(hf.states):
            n2, l2, *q2 = s2
            dq = [abs(q1[k] - q2[k]) for k in range(len(q1))]
            n_differ = sum(1 for d in dq if d != 0)
            is_valid = (abs(l1 - l2) == 1) and (n_differ == 1) and (max(dq) == 1)
            if not is_valid:
                assert M[i, j] == approx(0), f"Nonzero at ({s1}, {s2}): {M[i, j]}"


def test_hf_hermitian():
    hf = matrixHfNF(3, 5, [1, 1], defects=DEFECTS)
    for M in hf.matrix:
        assert M == approx(M.T)


def test_hf_call_hermitian():
    hf = matrixHfNF(3, 5, [1, 1], defects=DEFECTS)
    M = hf([2.3, 0.7])
    assert M == approx(M.T)


def test_hf_cache_efficiency():
    matrix_stark.clear_cache()
    hf = matrixHfNF(3, 5, [2, 2], defects=DEFECTS)
    info = matrix_stark.radial_integral.cache_info()
    n_atomic_pairs = sum(
        1 for n1, l1 in nlbasis(3, 5)
        for n2, l2 in nlbasis(3, 5)
        if abs(l1 - l2) == 1
    )
    assert info.misses <= n_atomic_pairs


# ---------------------------------------------------------------------------
# matrix ABC
# ---------------------------------------------------------------------------

def test_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        matrix()


def test_missing_generate_matrix_raises():
    with pytest.raises(TypeError):
        class Incomplete(matrix):
            def __init__(self):
                pass
        Incomplete()


# ---------------------------------------------------------------------------
# Tuple unpacking and __array__ guard
# ---------------------------------------------------------------------------

def test_hf_matrix_tuple_unpacking():
    hf = matrixHfNF(3, 5, [1, 1], defects=DEFECTS)
    m1, m2 = hf.matrix
    assert m1.shape == (hf.num_states, hf.num_states)
    assert m2.shape == (hf.num_states, hf.num_states)


def test_hf_np_asarray_raises():
    hf = matrixHfNF(3, 5, [1, 1], defects=DEFECTS)
    with pytest.raises(TypeError):
        np.asarray(hf)
