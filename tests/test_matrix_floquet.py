from .context import starktools
import numpy as np
from pytest import approx, raises

def test_basic_matrix_H0floquet():
    """Test object attributes were correctly initiated
    """
    nmin = 1
    nmax = 3
    q = 2
    freq = 1
    m = starktools.MatrixH0Floquet(nmin, nmax, q, freq)
    
    assert m.shape == (30, 30)
    assert m.q == 2
    assert m.nmax == 3
    assert m.nmin == 1
    assert m.frequency == 1

def test_matrix_H0Floquet():
    """Test structure of H0floquet matrix is correct.
    Should only have diagonal entries which follow E = -1/2n^2 + q*freq
    """
    nmin = 1
    nmax = 10
    q = 2
    freq = 2
    m = starktools.MatrixH0Floquet(nmin, nmax, q, freq)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q, q+1, dtype=np.int64)

    # Test diagonal entires are correct
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q in  qq:
                elm = m[n]
                assert m[n][l][q][n][l][q] == -0.5 * float(n)**(-2) + q * freq
    
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
    

def test_matrix_HsFloquet():
    """Test structure of HsFloquet matrix is correct.
    In the Floquet basis, the dc field only couples states for which m′= m, q′= q, and l' = l± 1.
    """
    nmin = 13
    nmax = 19
    q = 2
    freq = 2
    m = starktools.MatrixHsFloquet(nmin, nmax, q)
    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q, q+1, dtype=np.int64)

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



def test_matrix_HfFloquet():
    """Test structure of HfFloquet matrix is correct.
    In the Floquet basis, the time-dependent component of the field couples states for
    which q` = q ± 1, and l` = l ± 1
    """
    nmin = 13
    nmax = 19
    q = 2

    m = starktools.MatrixHfFloquet(nmin, nmax, q)

    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q, q+1, dtype=np.int64)

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


def test_matrix_HfFloquet_conversion():
    """Test structure of HfFloquet matrix is correct.
    In the Floquet basis, the time-dependent component of the field couples states for
    which q` = q ± 1, and l` = l ± 1
    """
    nmin = 13
    nmax = 19
    q = 1

    m = starktools.MatrixHfFloquet(nmin, nmax, q)

    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q, q+1, dtype=np.int64)

    array = np.array(m)
    # mm[13][0][0][13][1][1]
    i = 1
    j = 2 + 3
    assert(array[i][j] == m[13][0][0][13][1][1])


def test_matrix_HfFloquet_conversion():
    """Test structure of HfFloquet matrix is correct.
    In the Floquet basis, the time-dependent component of the field couples states for
    which q` = q ± 1, and l` = l ± 1
    """
    nmin = 13
    nmax = 19
    q = 2
    mm = starktools.MatrixHfFloquet(nmin, nmax, q)

    states = np.arange(min(mm.keys()), max(mm.keys())+1, 1) * (q*2 + 1)
    
    size = sum(states)
    lookuptable = np.empty((states, states),object)


    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    qq = np.arange(-q, q+1, dtype=np.int64)
    i = 0
    j = 0
    for a1 in mm.keys():
                for a2 in mm[a1].keys():
                    for a3 in mm[a1][a2].keys():
                        for b1 in mm[a1][a2][a3].keys():
                            for b2 in mm[a1][a2][a3][b1].keys():
                                for b3 in mm[a1][a2][a3][b1][b2].keys():
                                    lookuptable[i][j] = (a1,a2,a3,b1,b2,b3)
                                    j += 1
                        j = 0
                        i += 1
    
    
