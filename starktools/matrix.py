from functools import lru_cache
import numpy as np

class Matrix:
    def __init__(self, nmin: int, nmax: int):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['defects'] = {}

        self.matrix = self.gen_matrix_empty()

    @property # nmin should be read only
    def nmin(self):
        return self.__dict__['nmin']

    @property # nmax should be read only
    def nmax(self):
        return self.__dict__['nmax']

    @property # defects should be read only
    def defects(self):
        return self.__dict__['defects']

    def __array__(self, dtype=None):
        array = self.make_array()
        return array
    
    @property
    def shape(self):
        array = self.make_array()
        return array.shape
    
    def gen_matrix_empty(self):
        """
        Generate zero matrix of mm[n][l][n'][l']
        output is a nested dictionary
        """
        nn = np.arange(self.nmin, self.nmax+1, 1)
        mm = {}
        for a in nn:
            l1 = np.arange(0,a)
            m1 = {}
            for b in l1:
                m2 = {}
                for c in nn:
                    l2 = np.arange(0,c)
                    m3 = {}
                    for d in l2:
                        m3[d] = 0
                    m2[c] = m3
                m1[b] = m2
            mm[a] = m1
        return mm
    
    @lru_cache
    def make_array(self):
        mm = self.matrix

        size = sum(np.arange(min(mm.keys()), max(mm.keys())+1, 1))

        m = np.empty((size,size))
        i = 0
        j = 0

        for a in mm.keys():
            for b in mm[a].keys():
                for c in mm[a][b].keys():
                    for d in mm[a][b][c].keys():
                        m[i][j] = mm[a][b][c][d]
                        j += 1
                j = 0
                i += 1
        return m
    
    def _get_qd(self, n, l):
        qd = self.defects
        if l in qd:
            if type(qd[l]) == list:
                defect = 0
                m = n - qd[l][0]
                for i in range(len(qd[l])):
                    defect = defect + qd[l][i]*m**(-2*i)
                return defect
            else:
                return qd[l]
        else:
            return float(0)

    def __repr__(self):
        return str(self.matrix)

    def __mul__(self, scaler: float):

        mm = self.matrix
        matrixProduct = Matrix(self.nmin, self.nmax)
        for a in mm.keys():
            for b in mm[a].keys():
                for c in mm[a][b].keys():
                    for d  in mm[a][b][c].keys():
                        matrixProduct.matrix[a][b][c][d] = mm[a][b][c][d] * scaler
        return matrixProduct

    def __add__(self, other):
        mm = self.matrix
        matrixSum = Matrix(self.nmin, self.nmax)

        for a in mm.keys():
            for b in mm[a].keys():
                for c in mm[a][b].keys():
                    for d  in mm[a][b][c].keys():
                        matrixSum.matrix[a][b][c][d] = mm[a][b][c][d] + other.matrix[a][b][c][d]
        return matrixSum

    def __sub__(self, other):
        mm = self.matrix
        matrixSum = Matrix(self.nmin, self.nmax)

        for a in mm.keys():
            for b in mm[a].keys():
                for c in mm[a][b].keys():
                    for d  in mm[a][b][c].keys():
                        matrixSum.matrix[a][b][c][d] = mm[a][b][c][d] - other.matrix[a][b][c][d]
        return matrixSum
    
    def __getitem__(self, a):
        return self.matrix[a]