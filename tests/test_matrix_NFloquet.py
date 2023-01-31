from .context import starktools
import numpy as np
from pytest import approx, raises

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
    assert qq[0] == (-1, 0)

def test_nlqqbasis():
    nmin = 3
    nmax = 4
    qmaxs = [1,1]
    basis = []
    for n, l, q1, q2 in starktools.nlqqbasis(nmin, nmax, qmaxs):
        basis.append((n, l, q1, q2))
    assert basis[0] == (3, 0, -1, 0)

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

def test_compare_H0NFloquet_H0Floquet():
    nmin = 1
    nmax = 10
    q = 2
    freq = 2

    m0_Floquet  =  starktools.MatrixH0Floquet(nmin, nmax, q, freq)
    m0_NFloquet = starktools.MatrixH0NFloquet(nmin, nmax, [q], [freq])

    np.testing.assert_array_equal(m0_Floquet, m0_NFloquet)

def test_compare_HsNFloquet_HsFloquet():
    nmin = 1
    nmax = 10
    q = 2
    freq = 2

    mH_Floquet  =  starktools.MatrixHsFloquet(nmin, nmax, q)
    mH_NFloquet = starktools.MatrixHsNFloquet(nmin, nmax, [q])

    assert  np.all(np.asarray(mH_Floquet) == np.asarray(mH_NFloquet))

def test_compare_HfNFloquet_HfFloquet():
    nmin = 1
    nmax = 10
    q = 2
    freq = 2

    mf_Floquet  =  starktools.MatrixHfFloquet(nmin, nmax, q)
    mf_NFloquet = starktools.MatrixHfNFloquet(nmin, nmax, [q], 0)

    assert  np.all(np.asarray(mf_Floquet) == np.asarray(mf_NFloquet))



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

    m = starktools.MatrixHfNFloquet(nmin, nmax, q, fieldnum=0)

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
                            for q2 in qq:
                                if np.abs(q-q2) == 1:
                                    if m[n][l][q][n2][l2][q2] == 0:
                                        print(n,l,q,n2,l2,q2)
                                    assert m[n][l][q][n2][l2][q2] != 0