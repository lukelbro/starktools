import functools

import starktools
import numpy as np
import pytest
from pytest import approx, raises
import scipy

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


def _el(m, b1, b2, matrix=None):
    """Matrix element between basis states b1 and b2 (state tuples).

    Pass ``matrix`` to read from an assembled array (e.g. ``hf(famps)``) instead of ``m.matrix``.
    """
    matrix = m.matrix if matrix is None else matrix
    return matrix[m.lookuptable[tuple(b1)], m.lookuptable[tuple(b2)]]


def test_matrix_dimensions():
    nmin = 3
    nmax = 5
    q = [2,2,2]
    field = [1,1,1]
    freqs = [1,1,1]
    
    count = 0
    for b1 in starktools.nlqqbasis(nmin,nmax,q):
        count += 1
    
    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freqs, defects=DEFECTS))
    assert h0.shape == (count, count)

    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(field)
    assert hf.shape == (count, count)

    Hs = np.asarray(starktools.MatrixHsNFFloquet(nmin, nmax, q, defects=DEFECTS))
    assert Hs.shape == (count, count)


        
def test_matrix_H0NFloquet():
    """Test structure of H0floquet matrix is correct.
    Should only have diagonal entries which follow E = -1/2n^2 + q*freq
    """
    nmin = 2
    nmax = 10
    q = [2]
    freq = [2]
    m = starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q[0], q[0]+1, dtype=np.int64)

    # Test diagonal entires are correct
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in  qq:
                assert _el(m, (n, l, q), (n, l, q)) == approx(
                    QD.energy_level(n, l) + q * freq[0])
    
    # Test non diagonal entries are zero
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in qq:

                for n2 in nn:
                    ll2 = np.arange(0, n2, dtype=np.int64)
                    for l2 in ll2:
                        for q2 in qq:
                            if (n != n2) | (l != l2) | (q != q2):
                                assert _el(m, (n, l, q), (n2, l2, q2)) == 0

def test_H0NFloquet_sample_vals():
    h = starktools.MatrixH0NFFloquet(3,5,[1,1,], [4,3], defects=DEFECTS)
    h0 = starktools.matrixH0(3,5, defects=DEFECTS)
    e30 = h0.matrix[h0.lookuptable[(3, 0)], h0.lookuptable[(3, 0)]]
    assert _el(h, (3, 0, 1, 1), (3, 0, 1, 1)) == approx(e30 + 4 + 3, rel=1e-5)
    assert _el(h, (3, 0, 1, 0), (3, 0, 1, 0)) == approx(e30 + 4, rel=1e-5)
    assert _el(h, (3, 0, 0, 1), (3, 0, 0, 1)) == approx(e30 + 3, rel=1e-5)
    assert _el(h, (3, 0, 0, 1), (3, 0, 0, 0)) == 0

def test_HsNFloquet_sample_vals():
    h = starktools.MatrixHsNFFloquet(3,5,[1,1], defects=DEFECTS)
    qd = starktools.QuantumDefects(DEFECTS)
    assert (_el(h, (3, 1, 0, 0), (3, 0, 0, 0)) == approx(qd.calc_matrix_element(3, 1, 3, 0, nmax = 5), rel=1e-5))
    assert (_el(h, (3, 0, 0, 0), (3, 1, 0, 0)) == approx(qd.calc_matrix_element(3, 0, 3, 1, nmax = 5), rel=1e-5))
    assert (_el(h, (3, 0, 0, 0), (3, 1, 0, 0)) == _el(h, (3, 1, 0, 0), (3, 0, 0, 0)))

    assert (_el(h, (3, 0, 0, 1), (3, 1, 0, 1)) == approx(qd.calc_matrix_element(3, 0, 3, 1, nmax = 5), rel=1e-5))
    assert (_el(h, (3, 0, 0, 1), (3, 1, 0, 1)) == approx(qd.calc_matrix_element(3, 0, 3, 1, nmax = 5), rel=1e-5))
    assert (_el(h, (3, 0, 0, 1), (3, 1, 0, 1)) == approx(qd.calc_matrix_element(3, 0, 3, 1, nmax = 5), rel=1e-5))

def test_HfNFloquet_sample_vals():
    # Double field strength to account for 0.5*f
    hf_obj = starktools.MatrixHfNFFloquet(3,5,[1,1], defects=DEFECTS)
    hf = hf_obj([2,2])
    qd = starktools.QuantumDefects(DEFECTS)
    assert (_el(hf_obj, (3, 0, 0, 0), (3, 1, 1, 0), hf) == approx(qd.calc_matrix_element(3, 1, 3, 0, nmax = 5), rel=1e-5))
    assert (_el(hf_obj, (3, 1, 0, 0), (3, 0, 1, 0), hf) == approx(qd.calc_matrix_element(3, 0, 3, 1, nmax = 5), rel=1e-5))
    assert (_el(hf_obj, (3, 1, 1, 1), (3, 0, 1, 0), hf) == approx(qd.calc_matrix_element(3, 1, 3, 0, nmax = 5), rel=1e-5))
    
    assert (_el(hf_obj, (3, 1, 0, 1), (3, 0, 1, 0), hf) == 0)



def test_matrix_H0NFloquet_two_fields():
    """Test structure of H0floquet matrix is correct.
    Should only have diagonal entries which follow E = -1/2n^2 + q*freq
    """
    nmin = 2
    nmax = 10
    q = [2,1]
    freq = [2,1]

    m = starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq1 = np.arange(-q[0], q[0]+1, dtype=np.int64)
    qq2 = np.arange(-q[1], q[1]+1, dtype=np.int64)

    # Test diagonal entires are correct
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q1 in  qq1:
                for q2 in  qq2:
                    elm = _el(m, (n, l, q1, q2), (n, l, q1, q2))
                    val = QD.energy_level(n, l) + q1 * freq[0] + q2 * freq[1]
                    assert elm == approx(val)
    
    # Test non diagonal entries are zero
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for qa in qq1:
                for qb in qq2:
                    for n2 in nn:
                        ll2 = np.arange(0, n2, dtype=np.int64)
                        for l2 in ll2:
                            for qa2 in qq1:
                                for qb2 in qq2:
                                    if (n != n2) | (l != l2) | (qa != qa2) | (qb != qb2) :
                                        assert _el(m, (n, l, qa, qb), (n2, l2, qa2, qb2)) == 0

def test_matrix_H0NFloquet_two_fields_test_2():
    """Test structure of H0floquet matrix is correct.
    Should only have diagonal entries which follow E = -1/2n^2 + q*freq
    """
    nmin = 2
    nmax = 10
    q = [2,2]
    freq = [2,5]
    m = starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS)
    
    for n1, l1, qa1, qb1 in starktools.nlqqbasis(nmin, nmax, q):
        for n2, l2, qa2, qb2 in starktools.nlqqbasis(nmin, nmax, q):
            if (n1, l1, qa1, qb1) == (n2, l2, qa2, qb2):
                assert _el(m, (n1, l1, qa1, qb1), (n2, l2, qa2, qb2)) == approx(
                    QD.energy_level(n1, l1) + qa1 * freq[0] + qb1 * freq[1])
            else:
                assert _el(m, (n1, l1, qa1, qb1), (n2, l2, qa2, qb2)) == 0


def test_compare_H0NFloquet_H0Floquet():
    nmin = 2
    nmax = 10
    q = 2
    freq = 2

    m0_Floquet  =  starktools.matrixH0F(nmin, nmax, q, freq, defects=DEFECTS)
    m0_NFloquet = starktools.MatrixH0NFFloquet(nmin, nmax, [q], [freq], defects=DEFECTS)

    np.testing.assert_array_equal(np.asarray(m0_Floquet), np.asarray(m0_NFloquet))

def test_compare_HsNFloquet_HsFloquet():
    nmin = 2
    nmax = 10
    q = 2
    freq = 2

    mH_Floquet  =  starktools.matrixHsF(nmin, nmax, q, defects=DEFECTS)
    mH_NFloquet = starktools.MatrixHsNFFloquet(nmin, nmax, [q], defects=DEFECTS)

    np.testing.assert_array_equal(np.asarray(mH_Floquet), np.asarray(mH_NFloquet))

def test_compare_HfNFloquet_HfFloquet():
    nmin = 2
    nmax = 10
    q = 2
    freq = 2
    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }


    # Single-field wrapper is the N=1 case of the v2 N-field class; the unit
    # matrix (which carries the 1/2 sideband factor) must match bit-for-bit.
    mf_Floquet_unit = starktools.matrixHfF(nmin, nmax, q, defects=DEFECTS).matrix[0]
    mf_NFloquet_unit = starktools.MatrixHfNFFloquet(nmin, nmax, [q], defects=DEFECTS).matrix[0]

    np.testing.assert_array_equal(np.asarray(mf_Floquet_unit), np.asarray(mf_NFloquet_unit))
    # Assembling with an amplitude scales linearly: hf([f]) = f * unit.
    np.testing.assert_array_equal(starktools.matrixHfF(nmin, nmax, q, defects=DEFECTS)([2.0]), 2.0 * mf_Floquet_unit)



def test_matrix_HsNFloquet():
    """Test structure of HsFloquet matrix is correct.
    In the Floquet basis, the dc field only couples states for which m′= m, q′= q, and l' = l± 1.
    """
    nmin = 13
    nmax = 19
    q = [2]
    freq = [2]
    m =  starktools.MatrixHsNFFloquet(nmin, nmax, q, defects=DEFECTS)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    
    qq = np.arange(-q[0], q[0]+1, dtype=np.int64)

    # Test diagonal entires are zero
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in  qq:
                assert _el(m, (n, l, q), (n, l, q)) == 0
    
    # Check correct terms are coupled correctly
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for n2 in nn:
                ll2 = np.arange(0, n2, dtype=np.int64)
                for l2 in ll2:
                    if np.abs(l-l2) == 1:
                        for q in qq:
                            assert _el(m, (n, l, q), (n2, l2, q)) != 0
    
    # Check other entries are zero
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in qq:
                for n2 in nn:
                    ll2 = np.arange(0, n2, dtype=np.int64)
                    for l2 in ll2:
                        for q2 in qq:
                            if  (np.abs(l-l2) != 1):
                                assert _el(m, (n, l, q), (n2, l2, q2)) == 0

    # Check a value (numerov on defect-corrected effective quantum numbers)
    ns19 = 19 - QD.quantum_defect(19, 0)
    ns13 = 13 - QD.quantum_defect(13, 1)
    wf1 = starktools.Tools.numerov(ns19, 0, 19)
    wf2 = starktools.Tools.numerov(ns13, 1, 19)
    radint = starktools.Tools.numerov_calc_matrix_element(wf1, wf2)
    l2 = 1
    angularElem = (l2**2)/((2*l2+1)*(2*l2-1))
    angularElem = np.sqrt(angularElem)

    for q in qq:
        assert _el(m, (19, 0, q), (13, 1, q)) == approx(radint * angularElem)

def test_matrix_HfNFloquet():
    """Test structure of HfFloquet matrix is correct.
    In the Floquet basis, the time-dependent component of the field couples states for
    which q` = q ± 1, and l` = l ± 1
    """
    nmin = 13
    nmax = 19
    q = [2]

    hf_obj = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)
    hf = hf_obj([1])

    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q[0], q[0]+1, dtype=np.int64)

    # Test diagonal entires are zero
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in  qq:
                assert _el(hf_obj, (n, l, q), (n, l, q), hf) == 0
    
    # Check correct terms are coupled
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for n2 in nn:
                ll2 = np.arange(0, n2, dtype=np.int64)
                for l2 in ll2:
                    if np.abs(l-l2) == 1:
                        for q in qq:
                            for q2 in qq:
                                if np.abs(q-q2) == 1:
                                    if _el(hf_obj, (n, l, q), (n2, l2, q2), hf) == 0:
                                        print(n,l,q,n2,l2,q2)
                                    assert _el(hf_obj, (n, l, q), (n2, l2, q2), hf) != 0


def test_matrix_HfNFloquet_two_colors():
    """Test structure of HfFloquet matrix is correct.
    In the Floquet basis, the time-dependent component of the field couples states for
    which q` = q ± 1, and l` = l ± 1
    """

    @functools.cache
    def reference_element(n1, l1, n2, l2):
        return qd.calc_matrix_element(n1, l1, n2, l2, nmax)

    def matrix_elm(basis, f1, f2):
        n1, l1, qa1, qb1, n2, l2, qa2, qb2 = basis
        val = reference_element(n1, l1, n2, l2)

        elm = 0
        if np.abs(qa1-qa2) + np.abs(qb1-qb2) != 1:
            return elm
        if np.abs(qa1-qa2) == 1:
            elm += 0.5*val*f1
        if np.abs(qb1-qb2) == 1:
            elm += 0.5*val*f2
        return elm

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    qd = starktools.QuantumDefects(defects)


    nmin = 13
    nmax = 19
    q = (2,2)
    famps = (12, 0.1)

    hf_obj = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)
    hf = hf_obj(famps)


    for b1 in starktools.nlqqbasis(nmin, nmax, q):
        n1, l1, qa1, qb1 = b1
        for b2 in starktools.nlqqbasis(nmin, nmax, q):
            n2, l2, qa2, qb2 = b2
            if np.abs(l1-l2)==1:
                if np.abs(qa1-qa2) == 1 or np.abs(qb1-qb2) == 1:
                    elm = matrix_elm(b1+b2, famps[0], famps[1])
                    assert _el(hf_obj, (n1, l1, qa1, qb1), (n2, l2, qa2, qb2), hf) == approx(elm, rel=10e-10)


def test_init_NFlouet_with_single_ac_fields():
    nmin = 5
    nmax = 8
    q = [1]
    vac = [0.1]
    hfn = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)
    hfn(vac)

def test_init_NFlouet_with_two_ac_fields():
    nmin = 5
    nmax = 8
    q = [1,1]
    vac = [0.1, 0.1]
    hfn = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)
    hfn(vac)

def test_coupling_delta_m1_is_1_delta_m2_is_also_1():
    nmin = 2
    nmax = 3
    q = (2,2)
    f = (0.8,0.5)
    hf_obj = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)
    hf = hf_obj(f)

    n1 = 2
    l1 = 0
    ma1 = 1
    mb1 = 1
    n2 = 2
    l2 = 1
    ma2 = 1
    mb2 = 2
    
    assert (_el(hf_obj, (n1, l1, ma1, mb1), (n2, l2, ma2, mb2), hf) != 0)

    n1 = 2
    l1 = 0
    ma1 = 1
    mb1 = 1
    n2 = 2
    l2 = 1
    ma2 = 0
    mb2 = 2

    wf1  = starktools.Tools.numerov(n1, l1, nmax)
    wf2 = starktools.Tools.numerov(n2, l2, nmax)
    radint = starktools.Tools.numerov_calc_matrix_element(wf1, wf2)
    angularElem = (l2**2)/((2*l2+1)*(2*l2-1))
    angularElem = np.sqrt(angularElem)
    elm = radint * angularElem

    val = 0.5*elm*f[0] + 0.5*elm*f[1]
    assert (_el(hf_obj, (n1, l1, ma1, mb1), (n2, l2, ma2, mb2), hf) == 0)

    n1 = 2
    l1 = 0
    ma1 = 1
    mb1 = 1
    n2 = 2
    l2 = 1
    ma2 = 2
    mb2 = 2
    
    assert (_el(hf_obj, (n1, l1, ma1, mb1), (n2, l2, ma2, mb2), hf) == 0)


def test_ac_starkshift_one_ac_field1():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
        return find_nearest(v, energy)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    qd = starktools.QuantumDefects(defects)

    electricDipoleMoment = qd.calc_matrix_element(55, 0, 55, 1, nmax = 80) * starktools.Constants.e * starktools.Constants.a_He

    estrength = 0.1

    rabi = electricDipoleMoment * estrength/starktools.Constants.h

    assert rabi/2 == 1571849.9009367884
    nmin = 55
    nmax = 55
    q = [1]
    freq = [9.118568e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118568e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0.1]

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(vac) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=200)

def test_ac_starkshift_two_ac_field1():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
        return find_nearest(v, energy)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    qd = starktools.QuantumDefects(defects)

    electricDipoleMoment = qd.calc_matrix_element(55, 0, 55, 1, nmax = 80) * starktools.Constants.e * starktools.Constants.a_He

    estrength = 0.1

    rabi = electricDipoleMoment * estrength/starktools.Constants.h

    assert rabi/2 == 1571849.9009367884
    nmin = 54
    nmax = 58
    q = [1,1]
    freq = [9.118568e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118568e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0.1, 0]

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(vac) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=1000)
   
def test_ac_starkshift_two_ac_field2():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
        return find_nearest(v, energy)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    qd = starktools.QuantumDefects(defects)

    electricDipoleMoment = qd.calc_matrix_element(55, 0, 55, 1, nmax = 80) * starktools.Constants.e * starktools.Constants.a_He

    estrength = 0.1

    rabi= electricDipoleMoment * estrength/starktools.Constants.h


    assert rabi/2 == 1571849.9009367884

    nmin = 54
    nmax = 58
    q = [1,1]
    freq = [9.118568e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118568e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0, 0.1]

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(vac) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=1000)


def test_ac_starkshift_two_ac_field3():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
        return find_nearest(v, energy)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    qd = starktools.QuantumDefects(defects)

    electricDipoleMoment = qd.calc_matrix_element(55, 0, 55, 1, nmax = 80) * starktools.Constants.e * starktools.Constants.a_He

    estrength = 0.01

    rabi1 = electricDipoleMoment * estrength/starktools.Constants.h


    electricDipoleMoment = qd.calc_matrix_element(56, 0, 56, 1, nmax = 80) * starktools.Constants.e * starktools.Constants.a_He

    estrength = 0.01

    rabi2 = electricDipoleMoment * estrength/starktools.Constants.h

    nmin = 54
    nmax = 58
    q = [1,1]
    freq = [9.118568e9 * starktools.Constants.h /starktools.Constants.E_He, 8637175913* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0.01, 0.01]

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(vac) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi1/2)

    ind56s = find_eigen(56, 0, val, offset=-rabi2/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi1/2, abs=200)
    assert (-1060115473616.9417 - val[ind56s]) == approx(rabi2/2, abs=300)


def test_ac_starkshift_commensurate():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
        return find_nearest(v, energy)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    qd = starktools.QuantumDefects(defects)

    electricDipoleMoment = qd.calc_matrix_element(55, 0, 55, 1, nmax = 80) * starktools.Constants.e * starktools.Constants.a_He

    estrength = 0.1

    rabi = electricDipoleMoment * estrength/starktools.Constants.h

    assert rabi/2 == 1571849.9009367884
    nmin = 55
    nmax = 55
    q = [5,5]
    freq = [9.11857e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118566e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0.1/2, 0.1/2]

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(vac) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    M = (h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    val = scipy.linalg.eigvalsh(M)
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    # Will not work as frequencies are commenserate!
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=500000)



def test_dc_stark_shift():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
        return find_nearest(v, energy)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }
    
    nmin = 54
    nmax = 57
    q = (0,0)
    freq = (19.25691e9 * starktools.Constants.h /starktools.Constants.E_He, 9.148568e9* starktools.Constants.h /starktools.Constants.E_He)
    vac = (0, 0)

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(vac) * 1/starktools.Constants.F_He
    hs = np.asarray(starktools.MatrixHsNFFloquet(nmin, nmax, q, defects=DEFECTS)) * 1/starktools.Constants.F_He

    fmin = 0
    fmax = 10
    fsteps =  10
    field = np.linspace(fmin, fmax, fsteps)

    neig = h0.shape[0]
    
    vals = np.zeros((fsteps, neig))
    
    for i in range(fsteps):
        val = np.linalg.eigvalsh(h0 + hs*field[i] + hf)
        for j in range(neig):
            vals[i,j] = val[j]*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, vals[:,0])
    ind56s = find_eigen(56, 0, vals[:,0])

    transition1 = (vals[:,ind56s] - vals[:,ind55s])/2
    
    q = (1,1)
    freq = (19.25691e9 * starktools.Constants.h /starktools.Constants.E_He, 9.148568e9* starktools.Constants.h /starktools.Constants.E_He)
    vac = (0, 0)

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, q, freq, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, q, defects=DEFECTS)(vac) * 1/starktools.Constants.F_He
    hs = np.asarray(starktools.MatrixHsNFFloquet(nmin, nmax, q, defects=DEFECTS)) * 1/starktools.Constants.F_He

    fmin = 0
    fmax = 10
    fsteps =  10
    field = np.linspace(fmin, fmax, fsteps)

    neig = h0.shape[0]
    
    vals = np.zeros((fsteps, neig))
    
    for i in range(fsteps):
        val = np.linalg.eigvalsh(h0 + hs*field[i] + hf)
        for j in range(neig):
            vals[i,j] = val[j]*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, vals[:,0])
    ind56s = find_eigen(56, 0, vals[:,0])

    transition2 = (vals[:,ind56s] - vals[:,ind55s])/2

    assert (transition1 - transition2) == approx(0, abs=1e-3)


def test_shift_10_khz_ac_comparison_with_HF():
     
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
        return find_nearest(v, energy)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    defects = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]}
     
    nmin = 53
    nmax = 57
    qmaxs = [1,1]
    na = 55
    nb = 56
    fsteps = 10

    fmax1 = 0
    fmax2 = 0.1

    RESONANCE_sp = 9.118568e9
    RESONANCE_ss = 19.256910e9 #-calc_transition(na, 0, nb, 0)


    # HFN
    freq1 = (RESONANCE_sp+30e6)* starktools.Constants.h / starktools.Constants.E_He 
    freq2 = (RESONANCE_ss)* starktools.Constants.h / starktools.Constants.E_He
    freqs = (freq1, freq2)

    h0 = np.asarray(starktools.MatrixH0NFFloquet(nmin, nmax, qmaxs, freqs, defects=DEFECTS))
    hf = starktools.MatrixHfNFFloquet(nmin, nmax, qmaxs, defects=DEFECTS)((0, 0.1))

    vals = np.linalg.eigvals(h0 + hf/starktools.Constants.F_He)
    ind55s = find_eigen(na, 0, vals)
    ind56s = find_eigen(nb, 0, vals)

    shift_hfn = vals[ind56s]-vals[ind55s]- 39112998448.00716


    # HF (single-field; 1/2 sideband factor is inside the unit matrix)
    h0 = np.asarray(starktools.matrixH0F(nmin, nmax, 1, freqs[1], defects=DEFECTS))
    hf = starktools.matrixHfF(nmin, nmax, 1, defects=DEFECTS)
    vals = np.linalg.eigvals(h0 + hf([0.1]) / starktools.Constants.F_He)
    ind55s = find_eigen(na, 0, vals)
    ind56s = find_eigen(nb, 0, vals)

    shift_hf = vals[ind56s]-vals[ind55s]- 39112998448.00716

    assert shift_hf == approx(shift_hfn, abs=1e-10)
