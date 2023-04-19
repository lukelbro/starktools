from .matrix import Matrix
from .tools import Tools
from math import sqrt

class MatrixH0(Matrix):
    def __init__(self, nmin: int, nmax: int, defects = {}):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['defects'] = defects

        # Make H0 array
        mm = self.gen_matrix_empty()
        for i in mm.keys():
            for j in mm[i].keys():
                mm[i][j][i][j] = -0.5 * (i - self._get_qd(i,j))**(-2)
        self.matrix = mm

class MatrixHs(Matrix):
    def __init__(self, nmin: int, nmax: int, defects = {}):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
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

                    for j in mm[i][l1].keys():
                        if l2 < j:
                            ns2 = j - self._get_qd(j, l2)
                            wf2 = Tools.numerov(ns2, l2, self.nmax)
                            radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)

                            try:
                                test = mm[i][l1][j][l2] # Check for entry
                                angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
                                angularElem = sqrt(angularElem)
                                mm[i][l1][j][l2] = radialInt * angularElem
                            except KeyError:
                                pass

                            try:
                                test = mm[j][l2][i][l1]

                                angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
                                angularElem = sqrt(angularElem)
                                mm[j][l2][i][l1] = radialInt * angularElem
                            except KeyError:
                                pass
        self.matrix = mm