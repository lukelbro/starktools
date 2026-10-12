import starktools
import numpy as np
from pytest import approx, raises

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

def test_matrixH0F_attributes():
    nmin = 1
    nmax = 3
    q = 2
    freq = 1
    m = starktools.matrixH0F(nmin, nmax, q, freq)

    assert m.shape == (30, 30)
    assert m.qmaxs == [2]
    assert m.nmax == 3
    assert m.nmin == 1
    assert m.frequencies == [1]

def test_matrixH0F_diagonal():
    """Diagonal entries follow E = -1/(2 (n-delta)^2) + q*freq; off-diagonal are zero."""
    nmin = 2
    nmax = 10
    q = 2
    freq = 2
    m = starktools.matrixH0F(nmin, nmax, q, freq, defects=DEFECTS)

    for i, state in enumerate(m.states):
        n, l = state[0], state[1]
        qqv = state[2]
        assert m[i, i] == approx(QD.energy_level(n, l) + qqv * freq)

    off_diag = m.matrix.copy()
    np.fill_diagonal(off_diag, 0)
    assert np.all(off_diag == 0)

def test_matrixHsF_selection_rules():
    """DC field couples states with q' == q and |l' - l| == 1 only."""
    nmin = 13
    nmax = 19
    q = 2
    m = starktools.matrixHsF(nmin, nmax, q, defects=DEFECTS)
    qq = np.arange(-q, q + 1)

    # Diagonal is zero
    assert np.all(np.diag(m.matrix) == 0)

    for i, s1 in enumerate(m.states):
        n1, l1, q1 = s1
        for j, s2 in enumerate(m.states):
            n2, l2, q2 = s2
            if q1 == q2 and abs(l1 - l2) == 1:
                assert m[i, j] != 0
            elif q1 == q2:
                assert m[i, j] == 0

    # Check a specific value, including the quantum defect
    ns19 = 19 - QD.quantum_defect(19, 0)
    ns13 = 13 - QD.quantum_defect(13, 1)
    wf1 = starktools.Tools.numerov(ns19, 0, 19)
    wf2 = starktools.Tools.numerov(ns13, 1, 19)
    radint = starktools.Tools.numerov_calc_matrix_element(wf1, wf2)
    l2 = 1
    angularElem = np.sqrt((l2 ** 2) / ((2 * l2 + 1) * (2 * l2 - 1)))

    i = m.lookuptable[(19, 0, 0)]
    j = m.lookuptable[(13, 1, 0)]
    assert m[i, j] == approx(radint * angularElem)

def test_matrixHfF_selection_rules():
    """AC field couples states with |l' - l| == 1 and |q' - q| == 1 only."""
    nmin = 13
    nmax = 19
    q = 2
    hf = starktools.matrixHfF(nmin, nmax, q, defects=DEFECTS)
    m = hf.matrix[0]  # unit matrix (includes the 1/2 sideband factor)

    assert np.all(np.diag(m) == 0)

    for i, s1 in enumerate(hf.states):
        n1, l1, q1 = s1
        for j, s2 in enumerate(hf.states):
            n2, l2, q2 = s2
            if abs(l1 - l2) == 1 and abs(q1 - q2) == 1:
                assert m[i, j] != 0
            else:
                assert m[i, j] == 0

def test_matrixHfF_amplitude_linearity():
    """Assembling with an amplitude scales the unit matrix linearly."""
    hf = starktools.matrixHfF(13, 19, 2, defects=DEFECTS)
    unit = hf.matrix[0]
    f = 0.7
    np.testing.assert_allclose(hf([f]), f * unit)
    # multifield convention: hf([2.0]) carries the full (no extra 1/2) coupling
    np.testing.assert_allclose(hf([2.0]), 2.0 * unit)

def test_matrixHfF_asarray_raises():
    hf = starktools.matrixHfF(13, 19, 1, defects=DEFECTS)
    with raises(TypeError):
        np.asarray(hf)
