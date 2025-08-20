from .context import starktools
import numpy as np
from pytest import approx, raises


quantumdefect_fs = starktools.quantumdefect_fs

df_triplet = {
        0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
        1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
        2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
        3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
        4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
        5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
        6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
    }


def test_generate_atom_levels_fs():
    nmin = 2
    nmax = 5
    s = 1
    
    levels = list(starktools.basis_fs.generate_atom_levels_fs(nmin, nmax, s))
    
    num_levels = 0
    for n in range(nmin, nmax + 1):
        for l in range(n):
            for j in range(abs(l - s), l + s + 1):
                num_levels += 2*j + 1


    assert len(levels) == num_levels

    for n, l, S, j, m_j in levels:
        assert n >= nmin
        assert n <= nmax
        assert l >= 0
        assert l < n
        assert S == s
        assert j >= abs(l - S)
        assert j <= l + S
        assert m_j >= -j
        assert m_j <= j

    nmin = 4
    nmax = 7
    s = 1
    lmax = 2
    
    levels = list(starktools.basis_fs.generate_atom_levels_fs(nmin, nmax, s, lmax=lmax))
    
    num_levels = 0
    for n in range(nmin, nmax + 1):
        for l in range(lmax+1):
            for j in range(abs(l - s), l + s + 1):
                num_levels += 2*j + 1
    
    assert len(levels) == num_levels

    for n, l, S, j, m_j in levels:
        assert n >= nmin
        assert n <= nmax
        assert l >= 0
        assert l <= lmax
        assert S == s
        assert j >= abs(l - S)
        assert j <= l + S
        assert m_j >= -j
        assert m_j <= j


def test_compare_defect_triplet_vs_hf():
    n = 55 
    l = 0
    s = 1

    defects = quantumdefect_fs.calc_defect

    qd = starktools.QuantumDefects(df_triplet)
    defect_triplet = qd._get_qd(n, l)
    defect_fs = defects(n, l, s, 1)
    assert defect_triplet == approx(defect_fs)

    l = 1
    J = [0,1,2]
    dfs = []
    for j in J:
        defect_fs = defects(n, l, s, j)
        defect_fs 
        for i in range(2*j +1): #for weighting with mj
            dfs.append(defect_fs)
    assert qd._get_qd(n, l) == approx(np.mean(dfs), rel=1e-3)

    l = 2
    J = [1,2,3]
    dfs = []
    for j in J:
        defect_fs = defects(n, l, s, j)
        defect_fs 
        for i in range(2*j +1): #for weighting with mj
            dfs.append(defect_fs)
    
    assert qd._get_qd(n, l) == approx(np.mean(dfs), rel=1e-3)



##### TESTING
def test_compare_matrix_element_vs_triplets():
    n1 = 55
    n2 = 55
    l1 = 0
    l2 = 1
    mj1 = 1
    mj2 = 2
    s = 1
    j1 = 1
    j2 = 2
    q = 1
    nmax = 70

    matrix_fs = quantumdefect_fs.dipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, q, s, nmax)
    qd = starktools.QuantumDefects(df_triplet)
    matrix_triplet = qd.calc_matrix_element(n1, l1, n2, l2, nmax)

    
    assert matrix_fs == approx(matrix_triplet, rel=1e-2)
    


def test_compare_energy_level_vs_triplet():
    n = 55
    l = 0
    s = 1
    j = 1
    qd = starktools.QuantumDefects(df_triplet)
    energy_triplet = qd.energy_level(n, l)


    energy_fs = quantumdefect_fs.energylevel(n, l, s, j)
    assert energy_fs == approx(energy_triplet, rel=1e-30)

    n = 55
    l = 1
    s = 1
    j = 2
    qd = starktools.QuantumDefects(df_triplet)
    energy_triplet = qd.energy_level(n, l)
    energy_fs = quantumdefect_fs.energylevel(n, l, s, j)
    assert energy_fs == approx(energy_triplet, rel=1e-10)



def test_calc_transition():
    assert quantumdefect_fs.calc_transition(55, 0, 1, 1, 55, 1,1, 2) == approx(-9.118568, rel=1e-7)
    assert quantumdefect_fs.calc_transition(55, 0, 1, 1, 55, 1, 1, 1) == approx(-9.118568, rel=1e-4)
    assert quantumdefect_fs.calc_transition(55, 0, 1, 1, 55, 1, 1, 0) == approx(-9.118568, rel=1e-3)
test_calc_transition()

def test_calc_transitions():
    def calc_transition( n1, l1, n2, l2):
        qd = starktools.QuantumDefects(df_triplet)
        if (n1 == n2) and (l1==l2):
            return 0
        return starktools.Constants.c/(1/((qd.energy_level(n1, l1)*starktools.Constants.R_He  - qd.energy_level(n2, l2)*starktools.Constants.R_He)))*10**-9
    
    nmin = 52
    nmax = 58
    s = 1
    lmax = 2
    
    state = (55, 1, 1, 2, -2) 
    n1, l1, s1, j1, mj1 = state

    for level in starktools.basis_fs.generate_atom_levels_fs(nmin, nmax, s, lmax=lmax):
        n2, l2, s2, j2, mj2 = level
        if n1 != n2 or l1 != l2 or s1 != s2 or j1 != j2:
            transition_triplet = calc_transition(n1, l1, n2, l2)
            if transition_triplet == 0:
                pass
            else:
                assert calc_transition(n1, l1, n2, l2)  == approx(quantumdefect_fs.calc_transition(n1, l1, s1, j1, n2, l2, s2, j2), rel=1e-4)


def test_matrix_fshs():
    # Compare results with niave implementation
    class MatrixFsHs_poor(starktools.MatrixFs):
        def generate_matrix(self):
            matrix = np.zeros((self.num_states, self.num_states))
            for i, state1 in enumerate(self.states):
                n1, l1, s, j1, mj1 = state1
                for j, state2 in enumerate(self.states):        
                    n2, l2, s, j2, mj2 = state2
                    matrix[i, j] = starktools.quantumdefect_fs.dipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, self.polarization, s, self.nmax)
            return matrix
    
    M_poor = MatrixFsHs_poor(8,9,1,1)
    M = starktools.MatrixFsHs(8,9,1,1) 

    assert np.equal(M_poor.matrix, M.matrix).all()

def test_matrix_fs_H0_floquet():
    freq = 99
    M = starktools.MatrixFsH0Floquet(8,9,1,0,1,freq)
    for i in range(M.matrix.shape[0]):
        state1 = M.convert_index_to_state(i)
        n1, l1, s1, j1, mj1, q1 = state1
        for j in range(M.matrix.shape[1]):
            if i == j:
                assert M[i,j] == 0.5 * starktools.quantumdefect_fs.energylevel(n1, l1, s1, j1) + freq * q1
            else:
                assert M[i,j] == 0

def test_matrix_fs_Hf_floquet():
    freq = 99
    M = starktools.MatrixFsHfFloquet(8,9,1,0,1,freq)
    for i in range(M.matrix.shape[0]):
        state1 = M.convert_index_to_state(i)
        n1, l1, s1, j1, mj1, q1 = state1
        for j in range(M.matrix.shape[1]):
            state2 = M.convert_index_to_state(j)
            n2, l2, s2, j2, mj2, q2 = state2
            if np.abs(q1-q2)==1:
                assert M.matrix[i,j] == 0.5 * starktools.quantumdefect_fs.dipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, M.polarization, s1, M.nmax)
            else:
                assert M.matrix[i,j] == 0

def test_matrix_fs_Hs_floquet():
    freq = 99
    M = starktools.MatrixFsHsFloquet(8,9,1,0,1,freq)
    for i in range(M.matrix.shape[0]):
        state1 = M.convert_index_to_state(i)
        n1, l1, s1, j1, mj1, q1 = state1
        for j in range(M.matrix.shape[1]):
            state2 = M.convert_index_to_state(j)
            n2, l2, s2, j2, mj2, q2 = state2
            if np.abs(q1-q2)==0:
                assert M.matrix[i,j] ==  starktools.quantumdefect_fs.dipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, M.polarization, s1, M.nmax)
            else:
                assert M.matrix[i,j] == 0


def test_mjmax_feature():
    """Test the mjmax parameter for limiting magnetic quantum numbers in MatrixFsFloquet."""
    nmin, nmax = 5, 6
    S = 0.5  # spin-1/2
    qmax = 1
    frequency = 1.0
    polarization = 0
    
    # Test without mjmax (should include all possible mj values)
    matrix_no_limit = starktools.MatrixFsFloquet(nmin, nmax, S, polarization, qmax, frequency)
    states_no_limit = matrix_no_limit.states
    mj_values_no_limit = set(state[4] for state in states_no_limit)
    
    # Test with mjmax = 0.5 (should limit to |mj| <= 0.5)
    matrix_limited = starktools.MatrixFsFloquet(nmin, nmax, S, polarization, qmax, frequency, mjmax=0.5)
    states_limited = matrix_limited.states
    mj_values_limited = set(state[4] for state in states_limited)
    
    # Verify that all mj values in limited case are within the limit
    for state in states_limited:
        mj = state[4]
        assert abs(mj) <= 0.5, f"mj value {mj} exceeds limit of 0.5"
    
    # Verify that the limited case has fewer or equal states
    assert len(states_limited) <= len(states_no_limit)
    
    # Verify that all limited mj values are a subset of unlimited mj values
    assert mj_values_limited.issubset(mj_values_no_limit)
    
    # Test that mjmax=0.5 only includes mj values -0.5 and 0.5
    expected_mj_values = {-0.5, 0.5}
    assert mj_values_limited == expected_mj_values
    
    # Test the matrix functionality still works
    assert matrix_limited.matrix.shape[0] == len(states_limited)
    assert matrix_limited.matrix.shape[1] == len(states_limited)


def test_mjmax_feature_basis_generation():
    """Test mjmax parameter in the basis generation functions."""
    nmin, nmax = 3, 4
    S = 1  # triplet
    
    # Test without mjmax
    states_no_limit = list(starktools.basis_fs.generate_atom_levels_fs(nmin, nmax, S))
    mj_values_no_limit = set(state[4] for state in states_no_limit)
    
    # Test with mjmax = 1.5
    states_limited = list(starktools.basis_fs.generate_atom_levels_fs(nmin, nmax, S, mjmax=1.5))
    mj_values_limited = set(state[4] for state in states_limited)
    
    # Verify that all mj values in limited case are within the limit
    for state in states_limited:
        mj = state[4]
        assert abs(mj) <= 1.5, f"mj value {mj} exceeds limit of 1.5"
    
    # Test Floquet basis generation with mjmax
    qmax = 2
    floquet_states_limited = list(starktools.basis_fs.generate_atom_levels_fs_floquet(nmin, nmax, S, qmax, mjmax=1.5))
    
    for state in floquet_states_limited:
        mj = state[4]
        assert abs(mj) <= 1.5, f"mj value {mj} exceeds limit of 1.5"
    
    # Each regular state should appear (2*qmax + 1) times in Floquet basis
    expected_floquet_count = len(states_limited) * (2 * qmax + 1)
    assert len(floquet_states_limited) == expected_floquet_count