from nested_dict import nested_dict
from .matrix import Matrix
from .tools import Tools
from math import sqrt
import numpy as np
from functools import reduce
import operator
import itertools
from .basis import nlbasis, nlqbasis, nlqqbasis, qqbasis

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
            for b2 in nlqqbasis(self.nmin, self.nmax, self.qmaxs):
                m[i][j] = get_by_path(mm, b1 + b2)
                j+=1
            j = 0
            i +=1
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
            rybdergLevel = -0.5 * (n - self._get_qd(n,l))**(-2)
            for qq in qqbasis(self.qmaxs):
                value = rybdergLevel
                for i,  q in enumerate(qq):
                    value += self.frequencys[i] * q
                basis = ((n, l) + qq)
                element = basis + basis
                set_by_path(mm, element, value)
                
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

        qq = []
        for q1 in qqbasis(self.qmaxs):
            for q2 in qqbasis(self.qmaxs):
                    b1 = np.array(q1)
                    b2 = np.array(q2)
                    b = b1 - b2
                    if np.all(b == 0):
                        qq.append((q1, q2, np.where(b == 0)[0]))
    
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

                            for q1, q2, fieldid in qq:                                    
                                b1 = (n1, l1) + q1
                                b2 = (n2, l2) + q2
                                    
                                try: #entry may not exist?
                                    basis = b1 + b2
                                    set_by_path(mm, basis, value1)
                                except KeyError:
                                    pass

                                b1 = (n2, l2) + q1
                                b2 = (n1, l1) + q2
                                    
                                try: #entry may not exist?
                                    basis = b1 + b2
                                    set_by_path(mm, basis, value2)
                                except KeyError:
                                    pass


            self.matrix = mm

class MatrixHfNFloquet(MatrixNFloquet):
    def __init__(self, nmin: int, nmax: int, q: list, famps: list, defects = {}):
        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['qmaxs'] = q
        self.__dict__['defects'] = defects
        self.__dict__['famps'] = famps

        # Make HF array
        m = 0
        mm = self.gen_matrix_empty()

        elems = []
        for i in range(self.nmax-1):
            elem = (i, i+1)
            elems.append(elem)
        
        qq = []
        for q1 in qqbasis(self.qmaxs):
            for q2 in qqbasis(self.qmaxs):
                b1 = np.array(q1)
                b2 = np.array(q2)
                b = np.abs(b1 - b2)
                if np.sum(b) == 1:
                    fieldid = np.where(b == 1)[0]
                    if fieldid.size == 1:  #only couple if a single fourier side band is differnt by one
                        qq.append((q1, q2, fieldid))

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
                                                      
                            for qelement1, qelement2, fieldid in qq:
                                
                                    
                                v1 = 0
                                for id in fieldid:
                                    v1 += 0.5*value1*famps[int(id)]

                                b1 = (n1, l1) + tuple(qelement1)
                                b2 = (n2, l2) + tuple(qelement2)  
                                basis = b1 + b2
                                set_by_path(mm, basis, v1)


                                v2 = 0
                                for id in fieldid:
                                    v2 += 0.5*value2*famps[int(id)]

                                b1 = (n1, l1) + tuple(qelement1)
                                b2 = (n2, l2) + tuple(qelement2)  
                                basis = b2 + b1
                                set_by_path(mm, basis, v2)
                                

    
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

    def __iter__(self):
        """
        Generates every integer combination between [-a, -b, -c, -d ] and [a , b, c, d].
        
        Parameters:
        lst (list): List of n integers.
        
        Yields:
        tuple: A tuple of n integers representing a combination.
        
        Example:
        >>> list(integer_combinations([1, 2]))
        [(-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2)]
        """
        range_limits = [range(-x, x+1) for x in self.qmaxs]
        for combination in itertools.product(*range_limits):
            yield combination

class nlqqbasis:
    def __init__(self, nmin: int, nmax: int, qmaxs:list):
        self.nmin = nmin
        self.nmax = nmax
        self.qmaxs = qmaxs

    def __iter__(self):
        for n, l in nlbasis(self.nmin, self.nmax):
            for q in qqbasis(self.qmaxs):
                yield((n,l) +  q)
