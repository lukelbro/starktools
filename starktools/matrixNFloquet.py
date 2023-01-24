from nested_dict import nested_dict
from .matrix import Matrix
from .tools import Tools
from math import sqrt
import numpy as np
from functools import reduce
import operator
from nested_dict import nested_dict

class MatrixNFloquet(Matrix):
    def __init__(self, nmin: int, nmax: int, q: list):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['defects'] = {}
        self.__dict__['qmaxs'] = q
        
        self.matrix = self.gen_matrix_empty()

    def gen_matrix_empty(self):
        """
        Generate zero matrix of mm[n][l][q1][12].. [n'][l'][q1'][q2']
        output is a nested dictionary
        """
        self.__dict__['basislen'] =  2 + len(self.qmaxs)
        mm = nested_dict(self.basislen*2, float)


        for b1 in nlqqbasis(self.nmin, self.nmax, self.qmaxs):
            for b2 in nlqqbasis(self.nmin, self.nmax, self.qmaxs):
                element = b1 + b2
                set_by_path(mm, element, 0)
        
        return mm.to_dict()
    
    def make_array(self):
        mm = self.matrix
        states = np.arange(min(mm.keys()), max(mm.keys())+1, 1)
        # Account for oscillating fields
        for qmax in self.qmaxs:
            states = states * (2*qmax + 1)
        
        size = sum(states)
        m = np.empty((size,size))
        
        i = 0
        j = 0
        for b1 in nlqqbasis(self.nmin, self.nmax, self.qmaxs):
            j = 0
            i += 1
            for b2 in nlqqbasis(self.nmin, self.nmax, self.qmaxs):
                j+=1
        return m
    
class MatrixH0NFloquet(MatrixNFloquet):
    def __init__(self, nmin: int, nmax: int, q:list, frequencys:list, defects = {}):
        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['qmaxs'] = q
        self.__dict__['defects'] = defects
        self.__dict__['frequencys'] = frequencys

        # Make H0 array
        mm = self.gen_matrix_empty()
        for n, l in nlbasis(self.nmin, self.nmax):
            for i, qmax in enumerate(self.qmaxs):
                q = -qmax
                while q <= qmax:
                    value = -0.5 * (n - self._get_qd(n,l))**(-2) + q * self.frequencys[i]
                    qelement = np.zeros(len(self.qmaxs))
                    qelement[i] = q
                    basis = ((n, l) + tuple(qelement))
                    element = basis + basis
                    set_by_path(mm, element, value)
                    q += 1
        self.matrix = mm

class MatrixHsNFloquet(MatrixNFloquet):
    def __init__(self, nmin: int, nmax: int, q: list, defects = {}):
        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['qmaxs'] = q
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

            for n1 in mm.keys(): # n1
                if l1 < n1:
                    ns1 = n1 - self._get_qd(n1, l1)
                    wf1 = Tools.numerov(ns1, l1, self.nmax)

                    b1 = (n1, l1) + tuple([0]*len(self.qmaxs))
                    b2 = get_by_path(mm, b1)
                    
                    for n2 in b2.keys():
                        if l2 < n2:
                            ns2 = n2 - self._get_qd(n2, l2)
                            wf2 = Tools.numerov(ns2, l2, self.nmax)
                            radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)
                            
                            angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
                            angularElem = sqrt(angularElem)
                            value1 = radialInt * angularElem

                            angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
                            angularElem = sqrt(angularElem)
                            value2 = radialInt * angularElem
                            for i, qmax in enumerate(self.qmaxs):
                                q = -qmax
                                while q <= qmax:
                                    qelement = np.zeros(len(self.qmaxs))
                                    qelement[i] = q
                                    
                                    b1 = (n1, l1) + tuple(qelement)
                                    b2 = (n2, l2) + tuple(qelement)
                                    
                                    try: #entry may not exist?
                                        basis = b1 + b2
                                        set_by_path(mm, basis, value1)
                                        
                                        basis = b2 + b1
                                        set_by_path(mm, basis, value2)
                                    except KeyError:
                                        pass
                                    q += 1
            self.matrix = mm

class MatrixHfNFloquet(MatrixNFloquet):
    def __init__(self, nmin: int, nmax: int, q: list, defects = {}):
        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['qmaxs'] = q
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
        
            for n1 in mm.keys():
                if l1 < n1:
                    ns1 = n1 - self._get_qd(n1, l1)
                    wf1 = Tools.numerov(ns1, l1, self.nmax)

                    b1 = (n1, l1) + tuple([0]*len(self.qmaxs))
                    b2 = get_by_path(mm, b1)

                    for n2 in b2.keys():
                        if l2 < n2:
                            ns2 = n2 - self._get_qd(n2, l2)
                            wf2 = Tools.numerov(ns2, l2, self.nmax)
                            radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)
                            
                            angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
                            angularElem = sqrt(angularElem)
                            value1 = radialInt * angularElem

                            angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
                            angularElem = sqrt(angularElem)
                            value2 = radialInt * angularElem

                            for i, qmax in enumerate(self.qmaxs):
                                q = -qmax
                                while q <= qmax:
                                    qelement1 = np.zeros(len(self.qmaxs))
                                    qelement1[i] = q

                                    qelement2 = np.zeros(len(self.qmaxs))
                                    qelement2[i] = q + 1
      
                                    try:
                                        b1 = (n1, l1) + tuple(qelement1)
                                        b2 = (n2, l2) + tuple(qelement2)  
                                        basis = b1 + b2
                                        set_by_path(mm, basis, value1)
                                        
                                        b1 = (n1, l1) + tuple(qelement2)
                                        b2 = (n2, l2) + tuple(qelement1)  
                                        basis = b1 + b2
                                        set_by_path(mm, basis, value1)
                                    except KeyError:
                                        pass
                                    try:
                                        b1 = (n1, l1) + tuple(qelement1)
                                        b2 = (n2, l2) + tuple(qelement2)  
                                        basis = b2 + b1
                                        set_by_path(mm, basis, value2)

                                        b1 = (n1, l1) + tuple(qelement2)
                                        b2 = (n2, l2) + tuple(qelement1)  
                                        basis = b2 + b1
                                        set_by_path(mm, basis, value2)
                                    except KeyError:
                                        pass
                                    q += 1
        self.matrix = mm







def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)

def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value

class nlbasis:
    def __init__(self, nmin: int, nmax: int):
        self.nmin = nmin
        self.nmax = nmax
    
    def __iter__(self):
        n = self.nmin
        l = 0
        while n <= self.nmax:
            if l < n:
                yield (n, l)
                l += 1
            else:
                l = 0
                n += 1

class nlqbasis:
    def __init__(self, nmin: int, nmax: int, qmax:int):
        self.nmin = nmin
        self.nmax = nmax
        self.qmax = qmax
    
    def __iter__(self):
        for n, l in nlbasis(self.nmin, self.nmax):
            q = -self.qmax
            while q <= self.qmax:
                yield(n,l,q)
                q += 1

class qqbasis:
    def __init__(self, qmaxs:list):
        self.qmaxs = qmaxs
        self.qq = np.zeros(len(qmaxs))

    def __iter__(self):
        for i, qmax in enumerate(self.qmaxs):
            q = -qmax
            while q <= qmax:
                self.qq[i] = q
                yield(tuple(self.qq))
                q += 1
            self.qq[i] = 0

class nlqqbasis:
    def __init__(self, nmin: int, nmax: int, qmaxs:list):
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs

    def __iter__(self):
        for n, l in nlbasis(self.nmin, self.nmax):
            for q in qqbasis(self.qmaxs):
                yield((n,l) +  q)
