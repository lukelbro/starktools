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

def test_nlbasis_ml():
    nmin = 3
    nmax = 5
    ml = 1
    
    nl = []
    for n, l in starktools.nlbasis(nmin, nmax, ml):
        nl.append((n, l))
    
    assert nl[0] == (3, 1)
    assert nl[1] == (3, 2)
    assert nl[2] == (4, 1)
    assert nl[3] == (4, 2)
    assert nl[4] == (4, 3)
    assert nl[5] == (5, 1)
    assert nl[6] == (5, 2)
    assert nl[7] == (5, 3)
    assert nl[8] == (5, 4)

    ml = 2
    nl = []
    for n, l in starktools.nlbasis(nmin, nmax, ml):
        nl.append((n, l))
    
    assert nl[0] == (3, 2)
    assert nl[1] == (4, 2)
    assert nl[2] == (4, 3)
    assert nl[3] == (5, 2)
    assert nl[4] == (5, 3)
    assert nl[5] == (5, 4)


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
           
def test_nlqqbasis_explicit():
    nmin = 50
    nmax = 55
    qmaxs = [4,4]
    basis = []
    for n, l, q1, q2 in starktools.nlqqbasis(nmin, nmax, qmaxs):
        basis.append((n, l, q1, q2))

    basis = set(basis)

    nn = np.arange(nmin, nmax+1, dtype=np.int64)
    for n in nn:
        ll = np.arange(0, n, dtype=np.int64)
        for l in ll:
            for q1 in np.arange(-qmaxs[0], qmaxs[0]):
                for q2 in np.arange(-qmaxs[1], qmaxs[1]):
                    b1 = (n, l, q1, q2)
                    assert (b1 in basis) == True