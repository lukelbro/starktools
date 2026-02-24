from .context import starktools
import numpy as np
import pytest
from pytest import approx
from starktools.matrixNFFloquet_v2 import (
    MatrixH0NFFloquet,
    MatrixHsNFFloquet,
    MatrixHfNFFloquet,
    MatrixNFBase,
    quantumdefect_nl,
    DEFECTS,
)
from starktools.basis import nlbasis


# ---------------------------------------------------------------------------
# MatrixH0NFFloquet
# ---------------------------------------------------------------------------

def test_h0_diagonal_structure():
    nmin, nmax = 2, 4
    qmaxs = [1, 1]
    freqs = [3.0, 5.0]
    h0 = MatrixH0NFFloquet(nmin, nmax, qmaxs, freqs)

    for i, state in enumerate(h0.states):
        n, l = state[0], state[1]
        q_vec = state[2:]
        ns = n - quantumdefect_nl.calc_defect(n, l)
        expected = -0.5 * ns ** -2 + sum(freqs[k] * q_vec[k] for k in range(len(freqs)))
        assert h0.matrix[i, i] == approx(expected)

    off_diag = h0.matrix.copy()
    np.fill_diagonal(off_diag, 0)
    assert np.all(off_diag == 0)


def test_h0_matches_legacy():
    nmin, nmax = 3, 5
    qmaxs = [1, 1]
    freqs = [2.0, 1.5]
    h0_new = np.asarray(MatrixH0NFFloquet(nmin, nmax, qmaxs, freqs))
    h0_old = starktools.MatrixH0NFloquet(nmin, nmax, qmaxs, freqs, defects=DEFECTS).make_array()
    assert np.allclose(h0_new, h0_old, rtol=1e-5)


# ---------------------------------------------------------------------------
# MatrixHsNFFloquet
# ---------------------------------------------------------------------------

def test_hs_selection_rules():
    hs = MatrixHsNFFloquet(3, 5, [1, 1])
    for i, s1 in enumerate(hs.states):
        n1, l1, *q1 = s1
        for j, s2 in enumerate(hs.states):
            n2, l2, *q2 = s2
            if q1 != q2 or abs(l1 - l2) != 1:
                assert hs.matrix[i, j] == 0, f"Nonzero at ({s1}, {s2})"


def test_hs_values_match_quantumdefects():
    nmin, nmax = 3, 5
    hs = MatrixHsNFFloquet(nmin, nmax, [1, 1])
    qd = starktools.QuantumDefects(defects=DEFECTS)
    reference = qd.calc_matrix_element(3, 0, 3, 1, nmax=nmax)

    for q in range(-1, 2):
        for p in range(-1, 2):
            i = hs.lookuptable[(3, 0, q, p)]
            j = hs.lookuptable[(3, 1, q, p)]
            assert hs.matrix[i, j] == approx(reference, rel=1e-4)


def test_hs_hermitian():
    hs = MatrixHsNFFloquet(3, 5, [1, 1])
    assert hs.matrix == approx(hs.matrix.T)


# ---------------------------------------------------------------------------
# MatrixHfNFFloquet
# ---------------------------------------------------------------------------

def test_hf_num_unit_matrices():
    hf2 = MatrixHfNFFloquet(3, 5, [1, 1])
    assert len(hf2.matrix) == 2

    hf3 = MatrixHfNFFloquet(3, 5, [1, 1, 1])
    assert len(hf3.matrix) == 3


def test_hf_unit_matrices_disjoint():
    hf = MatrixHfNFFloquet(3, 5, [2, 2])
    masks = [M != 0 for M in hf.matrix]
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            assert not np.any(masks[a] & masks[b]), \
                f"Matrices {a} and {b} share nonzero elements"


def test_hf_call_matches_legacy():
    nmin, nmax = 3, 5
    qmaxs = [1, 1]
    famps = [2.0, 3.0]

    hf_new = MatrixHfNFFloquet(nmin, nmax, qmaxs)
    result_new = hf_new(famps)

    hf_old = starktools.MatrixHfNFloquet(nmin, nmax, qmaxs, famps, defects=DEFECTS).make_array()
    assert np.allclose(result_new, hf_old, rtol=1e-4)


def test_hf_call_linearity():
    hf = MatrixHfNFFloquet(3, 5, [1, 1])
    k = 3.7
    diff = hf([k, 1.0]) - hf([1.0, 1.0])
    expected = (k - 1) * hf.matrix[0]
    assert diff == approx(expected)


def test_hf_selection_rules():
    hf = MatrixHfNFFloquet(3, 5, [1, 1])
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
    hf = MatrixHfNFFloquet(3, 5, [1, 1])
    for M in hf.matrix:
        assert M == approx(M.T)


def test_hf_call_hermitian():
    hf = MatrixHfNFFloquet(3, 5, [1, 1])
    M = hf([2.3, 0.7])
    assert M == approx(M.T)


def test_hf_single_field_matches_legacy_single():
    nmin, nmax = 3, 5
    qmax = 2
    F = 1.5
    hf_new = MatrixHfNFFloquet(nmin, nmax, [qmax])
    result_new = hf_new([F])
    hf_old = starktools.MatrixHfNFloquet(nmin, nmax, [qmax], [F], defects=DEFECTS).make_array()
    assert np.allclose(result_new, hf_old, rtol=1e-4)


def test_hf_cache_efficiency():
    quantumdefect_nl.clear_cache()
    hf = MatrixHfNFFloquet(3, 5, [2, 2])
    info = quantumdefect_nl.radial_integral.cache_info()
    n_atomic_pairs = sum(
        1 for n1, l1 in nlbasis(3, 5)
        for n2, l2 in nlbasis(3, 5)
        if abs(l1 - l2) == 1
    )
    assert info.misses <= n_atomic_pairs


# ---------------------------------------------------------------------------
# MatrixNFBase ABC
# ---------------------------------------------------------------------------

def test_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        MatrixNFBase()


def test_missing_generate_matrix_raises():
    with pytest.raises(TypeError):
        class Incomplete(MatrixNFBase):
            def __init__(self):
                pass
        Incomplete()


# ---------------------------------------------------------------------------
# Tuple unpacking and __array__ guard
# ---------------------------------------------------------------------------

def test_hf_matrix_tuple_unpacking():
    hf = MatrixHfNFFloquet(3, 5, [1, 1])
    m1, m2 = hf.matrix
    assert m1.shape == (hf.num_states, hf.num_states)
    assert m2.shape == (hf.num_states, hf.num_states)


def test_hf_np_asarray_raises():
    hf = MatrixHfNFFloquet(3, 5, [1, 1])
    with pytest.raises(TypeError):
        np.asarray(hf)
