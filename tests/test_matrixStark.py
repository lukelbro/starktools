from .context import starktools
import numpy as np
from pytest import approx, raises

defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    
def test_matrixHs_comparison():
    h = starktools.MatrixHs(3,5)
    qd = starktools.QuantumDefects()
    assert (h[3][1][3][0] == qd.calc_matrix_element(3, 1, 3, 0, nmax = 5))
    assert (h[3][0][3][1] ==  qd.calc_matrix_element(3, 0, 3, 1, nmax = 5))
    assert (h[3][0][3][1] == h[3][1][3][0])

def test_matrixStarkmlH0():
    h = starktools.matrixmlH0(3,5, 1)
    states = h.states

    assert states[0] == (3, 1)
    assert states[1] == (3, 2)
    assert states[2] == (4, 1)
    assert states[3] == (4, 2)
    assert states[4] == (4, 3)
    assert states[5] == (5, 1)
    assert states[6] == (5, 2)
    assert states[7] == (5, 3)
    assert states[8] == (5, 4)
    assert h.num_states == 9

def test_matrixStarkmlHs_coupling_specific():
    nmin= 10
    nmax= 15
    h = starktools.matrixmlHs(nmin, nmax, 0, defects)
    state1 = 10, 0 
    state2 = 15, 1

    ind1 = h.lookuptable[state1]
    ind2 = h.lookuptable[state2]

    assert h[ind1][ind2] != 0

def test_matrixStarkmlHs_coupling():

    nmin = 10
    nmax = 15

    h = starktools.matrixmlHs(nmin, nmax, 0, defects)
    qd = starktools.QuantumDefects(defects)

    for state in h.states:
        n1, l1 = state
        ind1 = h.lookuptable[(n1, l1)]
        for state2 in h.states:
            n2, l2 = state2
            ind2 = h.lookuptable[(n2, l2)]
            if l2>l1: # In qd there is a very small numerical difference between the angular intergrals
                assert h[ind1][ind2] == qd.calc_matrix_element(n1, l1, n2, l2, nmax)
            if l2<l1:
                assert h[ind1][ind2] == qd.calc_matrix_element(n2, l2, n1, l1, nmax)

def test_compare_MatrixHs_matrixmlHs_basic():
    nmin = 52
    nmax = 54

    hs1 = starktools.MatrixHs(nmin, nmax, defects)
    hs2 = starktools.matrixmlHs(nmin, nmax, 0, defects)
    m1 = hs1.make_array()
    m2 = hs2.matrix
    assert m1.shape == m2.shape

    n1 = 52
    n2 = 53
    l1 = 0
    l2 = 1

    ind1 =  hs2.lookuptable[(n1, l1)]
    ind2 =  hs2.lookuptable[(n2, l2)]
    assert hs1[n1][l1][n2][l2] == m2[ind1][ind2]
    assert hs1[n2][l2][n1][l1] == m2[ind2][ind1]

def test_compare_MatrixHs_matrixmlHs_all():

    nmin = 52
    nmax = 54

    hs1 = starktools.MatrixHs(nmin, nmax, defects)
    hs2 = starktools.matrixmlHs(nmin, nmax, 0, defects)
    m1 = hs1.make_array()
    m2 = hs2.matrix

    np.testing.assert_array_equal(m1, m2)

def test_compare_MatrixH0_matrixmlH0_all():
    nmin = 10
    nmax = 13

    h01 = starktools.MatrixH0(nmin, nmax, defects)
    h02 = starktools.matrixmlH0(nmin, nmax, 0, defects)
    m1 = h01.make_array()
    m2 = h02.matrix

    np.testing.assert_array_equal(m1, m2)