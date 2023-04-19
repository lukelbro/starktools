from .context import starktools
import numpy as np
from pytest import approx, raises
import scipy


def test_nlbasis():
    """Test object attributes were correctly initiated
    """
    nmin = 3
    nmax = 5

    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    
    nlcompare = []
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            nlcompare.append((n,l))
    
    nl = []
    for n, l in starktools.nlbasis(nmin, nmax):
        nl.append((n, l))
    
    assert nl == nlcompare

def test_nlqbasis():
    """Test object attributes were correctly initiated
    """
    nmin = 3
    nmax = 5
    qmax = 1
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    
    nl = []
    for n, l, q in starktools.nlqbasis(nmin, nmax, qmax):
        nl.append((n, l, q ))
    
    assert nl[0] == (3, 0, -1)
    assert nl[1] == (3, 0, 0)
    assert nl[2] == (3, 0, 1)
    assert nl[3] == (3, 1, -1)
    assert nl[4] == (3, 1, 0)
    assert nl[5] == (3, 1, 1)

def test_qqbasis():
    qmaxs = [1,1]
    qq = []
    for q1, q2 in starktools.qqbasis(qmaxs):
        qq.append((q1, q2))
    assert qq[0] == (-1, -1)

def test_nlqqbasis():
    nmin = 3
    nmax = 5
    qmaxs = [1,1]
    basis = []
    for n, l, q1, q2 in starktools.nlqqbasis(nmin, nmax, qmaxs):
        basis.append((n, l, q1, q2))
    assert basis[0] == (3, 0, -1, -1)
    # check for value in the middle
    value1 = (3, 0, 1, 1)
    sucess = False
    for i in starktools.nlqqbasis(3,5,[1,1]):
        if i == value1:
            sucess = True
    assert sucess == True
           
def test_matrix_dimensions():
    nmin = 3
    nmax = 5
    q = [2,2,2]
    field = [1,1,1]
    freqs = [1,1,1]
    
    count = 0
    for b1 in starktools.nlqqbasis(nmin,nmax,q):
        count += 1
    
    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freqs))
    assert h0.shape == (count, count)

    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, field))
    assert hf.shape == (count, count)

    Hs = np.asarray(starktools.MatrixHsNFloquet(nmin, nmax, q))
    assert Hs.shape == (count, count)


        
def test_matrix_H0NFloquet():
    """Test structure of H0floquet matrix is correct.
    Should only have diagonal entries which follow E = -1/2n^2 + q*freq
    """
    nmin = 1
    nmax = 10
    q = [2]
    freq = [2]
    m = starktools.MatrixH0NFloquet(nmin, nmax, q, freq)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q[0], q[0]+1, dtype=np.int64)

    # Test diagonal entires are correct
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in  qq:
                elm = m[n]
                assert m[n][l][q][n][l][q] == -0.5 * float(n)**(-2) + q * freq[0]
    
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
                                assert m[n][l][q][n2][l2][q2] == 0

def test_H0NFloquet_sample_vals():
    h = starktools.MatrixH0NFloquet(3,5,[1,1,], [4,3])
    h0 = starktools.MatrixH0(3,5)
    assert h[3][0][1][1][3][0][1][1] == h0[3][0][3][0] + 4 + 3
    assert h[3][0][1][0][3][0][1][0] == h0[3][0][3][0] + 4
    assert h[3][0][0][1][3][0][0][1] == h0[3][0][3][0] + 3
    assert h[3][0][0][1][3][0][0][0] == 0

def test_HsNFloquet_sample_vals():
    h = starktools.MatrixHsNFloquet(3,5,[1,1])
    qd = starktools.QuantumDefects()
    assert (h[3][1][0][0][3][0][0][0] == qd.calc_matrix_element(3, 1, 3, 0, nmax = 5))
    assert (h[3][0][0][0][3][1][0][0] ==  qd.calc_matrix_element(3, 0, 3, 1, nmax = 5))
    assert (h[3][0][0][0][3][1][0][0] == h[3][1][0][0][3][0][0][0])

    assert (h[3][0][0][1][3][1][0][1] ==  qd.calc_matrix_element(3, 0, 3, 1, nmax = 5))
    assert (h[3][0][0][1][3][1][0][1] ==  qd.calc_matrix_element(3, 0, 3, 1, nmax = 5))
    assert (h[3][0][0][1][3][1][0][1] ==  qd.calc_matrix_element(3, 0, 3, 1, nmax = 5))

def test_HfNFloquet_sample_vals():
    # Double field strength to account for 0.5*f
    h = starktools.MatrixHfNFloquet(3,5,[1,1], [2,2])
    qd = starktools.QuantumDefects()
    assert (h[3][0][0][0][3][1][1][0] == qd.calc_matrix_element(3, 1, 3, 0, nmax = 5))
    assert (h[3][1][0][0][3][0][1][0] == qd.calc_matrix_element(3, 0, 3, 1, nmax = 5))
    assert (h[3][1][1][1][3][0][1][0] == qd.calc_matrix_element(3, 1, 3, 0, nmax = 5))
    
    assert (h[3][1][0][1][3][0][1][0] == 2*qd.calc_matrix_element(3, 1, 3, 0, nmax = 5))



def test_matrix_H0NFloquet_two_fields():
    """Test structure of H0floquet matrix is correct.
    Should only have diagonal entries which follow E = -1/2n^2 + q*freq
    """
    nmin = 1
    nmax = 10
    q = [2,1]
    freq = [2,1]

    m = starktools.MatrixH0NFloquet(nmin, nmax, q, freq)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq1 = np.arange(-q[0], q[0]+1, dtype=np.int64)
    qq2 = np.arange(-q[1], q[1]+1, dtype=np.int64)

    # Test diagonal entires are correct
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q1 in  qq1:
                for q2 in  qq2:
                    elm = m[n][l][q1][q2][n][l][q1][q2]
                    val = -0.5 * float(n)**(-2) + q1 * freq[0] + q2 * freq[1]
                    assert elm  == val
    
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
                                        assert m[n][l][qa][qb][n2][l2][qa2][qb2] == 0

def test_compare_H0NFloquet_H0Floquet():
    nmin = 1
    nmax = 10
    q = 2
    freq = 2

    m0_Floquet  =  starktools.MatrixH0Floquet(nmin, nmax, q, freq)
    m0_NFloquet = starktools.MatrixH0NFloquet(nmin, nmax, [q], [freq])

    np.testing.assert_array_equal(m0_Floquet, m0_NFloquet)

# def test__H0NFloquet_H0Floquet():
#     nmin = 1
#     nmax = 10
#     q = (2,1)
#     freq = (2,0.1)

#     m0_Floquet  =  starktools.MatrixH0Floquet(nmin, nmax, q, freq)
#     m0_NFloquet = starktools.MatrixH0NFloquet(nmin, nmax, q, freq)

#     np.testing.assert_array_equal(m0_Floquet, m0_NFloquet)

def test_compare_HsNFloquet_HsFloquet():
    nmin = 1
    nmax = 10
    q = 2
    freq = 2

    mH_Floquet  =  starktools.MatrixHsFloquet(nmin, nmax, q)
    mH_NFloquet = starktools.MatrixHsNFloquet(nmin, nmax, [q])

    np.testing.assert_array_equal(np.asarray(mH_Floquet), np.asarray(mH_NFloquet))

def test_compare_HfNFloquet_HfFloquet():
    nmin = 1
    nmax = 10
    q = 2
    freq = 2

    mf_Floquet  = np.asarray(starktools.MatrixHfFloquet(nmin, nmax, q))
    mf_NFloquet = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, [q], [1]))*2

    np.testing.assert_array_equal(mf_Floquet, mf_NFloquet)



def test_matrix_HsNFloquet():
    """Test structure of HsFloquet matrix is correct.
    In the Floquet basis, the dc field only couples states for which m′= m, q′= q, and l' = l± 1.
    """
    nmin = 13
    nmax = 19
    q = [2]
    freq = [2]
    m =  starktools.MatrixHsNFloquet(nmin, nmax, q)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    
    qq = np.arange(-q[0], q[0]+1, dtype=np.int64)

    # Test diagonal entires are zero
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in  qq:
                assert m[n][l][q][n][l][q] == 0
    
    # Check correct terms are coupled correctly
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for n2 in nn:
                ll2 = np.arange(0, n2, dtype=np.int64)
                for l2 in ll2:
                    if np.abs(l-l2) == 1:
                        for q in qq:
                            assert m[n][l][q][n2][l2][q] != 0
    
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
                                assert m[n][l][q][n2][l2][q2] == 0

    # Check a value
    wf1  = starktools.Tools.numerov(19, 0, 19)
    wf2 = starktools.Tools.numerov(13, 1, 19)
    radint = starktools.Tools.numerov_calc_matrix_element(wf1, wf2)
    l2 = 1
    angularElem = (l2**2)/((2*l2+1)*(2*l2-1))
    angularElem = np.sqrt(angularElem)

    for q in qq:
        assert m[19][0][q][13][1][q] == radint * angularElem

def test_matrix_HfNFloquet():
    """Test structure of HfFloquet matrix is correct.
    In the Floquet basis, the time-dependent component of the field couples states for
    which q` = q ± 1, and l` = l ± 1
    """
    nmin = 13
    nmax = 19
    q = [2]

    m = starktools.MatrixHfNFloquet(nmin, nmax, q, famps=[1])

    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q[0], q[0]+1, dtype=np.int64)

    # Test diagonal entires are zero
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in  qq:
                assert m[n][l][q][n][l][q] == 0
    
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
                                    if m[n][l][q][n2][l2][q2] == 0:
                                        print(n,l,q,n2,l2,q2)
                                    assert m[n][l][q][n2][l2][q2] != 0

def test_init_NFlouet_with_single_ac_fields():
    nmin = 5
    nmax = 8
    q = [1]
    vac = [0.1]
    hfn = starktools.MatrixHfNFloquet(nmin, nmax, q, vac)

def test_init_NFlouet_with_two_ac_fields():
    nmin = 5
    nmax = 8
    q = [1,1]
    vac = [0.1, 0.1]
    hfn = starktools.MatrixHfNFloquet(nmin, nmax, q, vac)

def test_coupling_delta_m1_is_1_delta_m2_is_also_1():
    nmin = 1
    nmax = 3
    q = (2,2)
    f = (0.8,0.5)
    h = starktools.MatrixHfNFloquet(nmin, nmax, q, f)

    n1 = 2
    l1 = 0
    ma1 = 1
    mb1 = 1
    n2 = 2
    l2 = 1
    ma2 = 1
    mb2 = 2
    
    assert (h[n1][l1][ma1][mb1][n2][l2][ma2][mb2] != 0)

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
    assert (h[n1][l1][ma1][mb1][n2][l2][ma2][mb2] == val)

    n1 = 2
    l1 = 0
    ma1 = 1
    mb1 = 1
    n2 = 2
    l2 = 1
    ma2 = 2
    mb2 = 2
    
    assert (h[n1][l1][ma1][mb1][n2][l2][ma2][mb2] == val)


def test_ac_starkshift_one_ac_field1():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h/2 + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
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

    assert rabi/2 == 1571675.449592038
    nmin = 55
    nmax = 55
    q = [1]
    freq = [9.118568e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118568e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0.1]

    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freq, defects))
    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, vac, defects)) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=200)

def test_ac_starkshift_two_ac_field1():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h/2 + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
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

    assert rabi/2 == 1571675.449592038
    nmin = 54
    nmax = 58
    q = [1,1]
    freq = [9.118568e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118568e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0.1, 0]

    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freq, defects))
    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, vac, defects)) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=1000)
   
def test_ac_starkshift_two_ac_field2():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h/2 + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
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


    assert rabi/2 == 1571675.449592038

    nmin = 54
    nmax = 58
    q = [1,1]
    freq = [9.118568e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118568e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0, 0.1]

    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freq, defects))
    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, vac, defects)) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=1000)


def test_ac_starkshift_two_ac_field3():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h/2 + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
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

    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freq, defects))
    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, vac, defects)) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    val = np.linalg.eigvalsh(h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    
    ind55s = find_eigen(55, 0, val, offset=-rabi1/2)

    ind56s = find_eigen(56, 0, val, offset=-rabi2/2)
    
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi1/2, abs=200)
    assert (-1060115473616.9417 - val[ind56s]) == approx(rabi2/2, abs=300)


def test_ac_starkshift_three_ac_fields():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h/2 + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
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

    assert rabi/2 == 1571675.449592038
    nmin = 55
    nmax = 55
    q = [5,5]
    freq = [9.11857e9 * starktools.Constants.h /starktools.Constants.E_He, 9.118566e9* starktools.Constants.h /starktools.Constants.E_He]
    vac = [0.1/2, 0.1/2]

    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freq, defects))
    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, vac, defects)) * 1/starktools.Constants.F_He 
    
    neig = h0.shape[0]
    M = (h0 + hf)*starktools.Constants.E_He/starktools.Constants.h
    val = scipy.linalg.eigvalsh(M)
    ind55s = find_eigen(55, 0, val, offset=-rabi/2)
    
    # Will not work as frequencies are commenserate!
    assert (-1099228472064.9489  - val[ind55s]) == approx(rabi/2, abs=1000)



def test_dc_stark_shift():
    
    def find_eigen(n, l, v, offset=0):
        qd = starktools.QuantumDefects(defects)
        energy = qd.energy_level(n, l) * starktools.Constants.E_He/starktools.Constants.h/2 + offset#+  qd.calc_matrix_element(55,0, 55, 1, 70)+offset
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

    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freq, defects))
    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, vac, defects)) * 1/starktools.Constants.F_He
    hs = np.asarray(starktools.MatrixHsNFloquet(nmin, nmax, q, defects)) * 1/starktools.Constants.F_He

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

    h0 = np.asarray(starktools.MatrixH0NFloquet(nmin, nmax, q, freq, defects))
    hf = np.asarray(starktools.MatrixHfNFloquet(nmin, nmax, q, vac, defects)) * 1/starktools.Constants.F_He
    hs = np.asarray(starktools.MatrixHsNFloquet(nmin, nmax, q, defects)) * 1/starktools.Constants.F_He

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