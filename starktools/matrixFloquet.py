from .matrix import Matrix
from .tools import Tools
from math import sqrt
import numpy as np

class MatrixFloquet(Matrix):
    def __init__(self, nmin: int, nmax: int, q: int):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['q'] = q
        self.__dict__['defects'] = {}

        self.matrix = self.gen_matrix_empty()

    def gen_matrix_empty(self):
        """
        Generate zero matrix of mm[n][l][q][n'][l'][q']
        output is a nested dictionary
        """
        nn = np.arange(self.nmin, self.nmax+1)
        qq = np.arange(-self.q, self.q+1)
        mm = {}

        for a1 in nn:
            l1 = np.arange(0,a1)
            m1 = {}
            for a2 in l1:
                m2 = {}
                for a3 in qq:
                    m3 = {}

                    for b1 in nn:
                        l2 = np.arange(0,b1)
                        m4 = {}
                        for b2 in l2:
                            m5 = {}
                            for b3 in qq:
                                m5[b3] = 0
                            m4[b2] = m5
                        m3[b1] = m4
                    m2[a3] = m3
                m1[a2] = m2
            mm[a1] = m1
        return mm

    def make_array(self):
        mm = self.matrix

        states = np.arange(min(mm.keys()), max(mm.keys())+1, 1) * (self.q*2 + 1)
        size = sum(states)

        m = np.empty((size,size))
        i = 0
        j = 0

        for a1 in mm.keys():
            for a2 in mm[a1].keys():
                for a3 in mm[a1][a2].keys():
                    for b1 in mm[a1][a2][a3].keys():
                        for b2 in mm[a1][a2][a3][b1].keys():
                            for b3 in mm[a1][a2][a3][b1][b2].keys():
                                m[i][j] = mm[a1][a2][a3][b1][b2][b3]
                                j += 1
                    j = 0
                    i += 1
        return m

class MatrixH0Floquet(MatrixFloquet):
    def __init__(self, nmin: int, nmax: int, q:int, frequency:float, defects = {}):
        """initializer for diagnol elements of stark matrix with floquet expansion.
        Args:
            nmin (int): minimum principle quantum number
            nmax (int): maximum principle quantum number
            q (int):  number of side bands to include in basis
            frequency (float): frequency of side bands in atomic units (freq_atomic = freq_Hz * h /E_He)
            defects (dict, optional): Dictionary of quantum defects. Defaults to {}.

        Raises:
            ValueError: The minimum quantum number should be less than the maximum
        """

        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['q'] = q
        self.__dict__['defects'] = defects
        self.__dict__['frequency'] = frequency

        # Make H0 array
        mm = self.gen_matrix_empty()
        for i in mm.keys(): 
            for j in mm[i].keys():
                for k in mm[i][j].keys():
                    mm[i][j][k][i][j][k] = -0.5 * (i - self._get_qd(i,j))**(-2) + k * self.frequency
        self.matrix = mm

class MatrixHsFloquet(MatrixFloquet):
    def __init__(self, nmin: int, nmax: int, q: int, defects = {}):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['q'] = q
        self.__dict__['defects'] = defects

        # Make HS array
        m = 0
        mm = self.gen_matrix_empty()
        elems = []
        for i in range(self.nmax-1):
            elem = (i, i+1)
            elems.append(elem)

        for elem in elems:
            l1 = elem[0]
            l2 = elem[1]

            for i in mm.keys(): # n1
                if l1 < i:
                    ns1 = i - self._get_qd(i, l1)
                    wf1 = Tools.numerov(ns1, l1, self.nmax)

                    for j in mm[i][l1][0].keys():
                        if l2 < j:
                            ns2 = j - self._get_qd(j, l2)
                            wf2 = Tools.numerov(ns2, l2, self.nmax)
                            radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)

                            try:
                                test = mm[i][l1][0][j][l2][0] # Check for entry

                                
                                angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
                                angularElem = sqrt(angularElem)

                                for k in mm[i][l1].keys():
                                    mm[i][l1][k][j][l2][k] = radialInt * angularElem
                            except KeyError:
                                pass

                            try:
                                test = mm[j][l2][0][i][l1][0]

                                angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
                                angularElem = sqrt(angularElem)
                                for k in mm[j][l2].keys():
                                    mm[j][l2][k][i][l1][k] = radialInt * angularElem
                            except KeyError:
                                pass
        self.matrix = mm

class MatrixHfFloquet(MatrixFloquet):
    def __init__(self, nmin: int, nmax: int, q: int, defects = {}):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['q'] = q
        self.__dict__['defects'] = defects

        # Make HF array
        m = 0
        mm = self.gen_matrix_empty()
        elems = []
        for i in range(self.nmax-1):
            elem = (i, i+1)
            elems.append(elem)

        for elem in elems:
            l1 = elem[0]
            l2 = elem[1]

            for i in mm.keys(): # n1
                if l1 < i:
                    ns1 = i - self._get_qd(i, l1)
                    wf1 = Tools.numerov(ns1, l1, self.nmax)

                    for j in mm[i][l1][0].keys():
                        if l2 < j:
                            ns2 = j - self._get_qd(j, l2)
                            wf2 = Tools.numerov(ns2, l2, self.nmax)
                            radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)

                            try:
                                test = mm[i][l1][0][j][l2][0] # Check for entry

                                angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
                                angularElem = sqrt(angularElem)

                                for k in mm[i][l1].keys():
                                    qi = k
                                    qj = qi + 1
                                    if (qj <= self.q):
                                        
                                        mm[i][l1][qi][j][l2][qj] = radialInt * angularElem
                                        mm[i][l1][qj][j][l2][qi] = radialInt * angularElem
                            except KeyError:
                                pass

                            try:
                                test = mm[j][l2][0][i][l1][0]

                                angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
                                angularElem = sqrt(angularElem)
                                for k in mm[j][l2].keys():
                                    qi = k
                                    qj = qi + 1
                                    if (qj <= self.q):
                                        mm[j][l2][qi][i][l1][qj] = radialInt * angularElem
                                        mm[j][l2][qj][i][l1][qi] = radialInt * angularElem
                            except KeyError:
                                pass
        self.matrix = mm