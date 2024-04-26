from nested_dict import nested_dict
from .matrix import Matrix
from .tools import Tools
from math import sqrt
import numpy as np
from functools import reduce, cache
import operator
import itertools
from .basis import nlbasis, nlqbasis, nlqqbasis, qqbasis

class matrix_v2:
    def __init__(self, nmin: int, nmax: int, defects = {}):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['defects'] = defects

        self.basis = nlbasis(self.nmin, self.nmax)
        self.states, self.lookuptable = self.generate_matrix_states()
        self.num_states = len(self.states)
        self.matrix = self.generate_matrix()
    
    @property # nmin should be read only
    def nmin(self):
        return self.__dict__['nmin']
    @property # nmax should be read only
    def nmax(self):
        return self.__dict__['nmax']
    @property # defects should be read only
    def defects(self):
        return self.__dict__['defects']
    
    def generate_matrix_states(self):
        states = []
        lookuptable = {}
        index = 0
        for state in self.basis:
            states.append(state)
            lookuptable[state] = index
            index += 1
        return states, lookuptable
    
    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        return matrix
    
    def convert_state_to_index(self, state):
        self.basis.state_valid_raise_exception(state)
        return self.lookuptable[state]

    def convert_states_to_index(self, state1, state2):
        self.basis.state_valid_raise_exception(state1)
        self.basis.state_valid_raise_exception(state2)
        return self.lookuptable[state1], self.lookuptable[state2]

    def convert_index_to_state(self, index):
        return self.states[index]

    def convert_indexs_to_states(self, index1, index2):
        return self.states[index1], self.states[index2]
        
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
    
    def __getitem__(self, index):
        return self.matrix[index]
    

class matrixmlH0(matrix_v2):
    def __init__(self, nmin: int, nmax: int, ml: int, defects = {}):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['ml'] = ml
        self.__dict__['defects'] = defects


        self.basis = nlbasis(self.nmin, self.nmax, self.ml)
        self.states, self.lookuptable = self.generate_matrix_states()
        self.num_states = len(self.states)
        self.matrix = self.generate_matrix()

    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
                n, l = self.states[i]
                matrix[i,i] = -0.5 * (n - self._get_qd(n,l))**(-2)
        return matrix


class matrixmlHs(matrix_v2):
    def __init__(self, nmin: int, nmax: int, ml: int, defects = {}):
        if nmin > nmax:
            raise ValueError('nmax should be greater than nmin')

        self.__dict__['nmin'] = nmin
        self.__dict__['nmax'] = nmax
        self.__dict__['ml'] = ml
        self.__dict__['defects'] = defects


        self.basis = nlbasis(self.nmin, self.nmax, self.ml)
        self.states, self.lookuptable = self.generate_matrix_states()
        self.num_states = len(self.states)
        self.matrix = self.generate_matrix()

    def generate_matrix(self):
        matrix = np.zeros((self.num_states, self.num_states))
        
        elems = []
        for i in range(np.abs(self.ml), self.nmax-1):
            elem = (i, i+1)
            elems.append(elem)

        for elem in elems:
            l1, l2 = elem
            for n1 in range(self.nmin, self.nmax+1):
                if l1 < n1:
                    ns1 = n1 - self._get_qd(n1, l1)
                    wf1 = Tools.numerov(ns1, l1, self.nmax)
                    for n2 in range(self.nmin, self.nmax+1):
                        if l2 < n2:
                            ns2 = n2 - self._get_qd(n2, l2)
                            wf2 = Tools.numerov(ns2, l2, self.nmax)
                            radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)
                            
                            ind1 = self.lookuptable[(n1, l1)]
                            ind2 = self.lookuptable[(n2, l2)]
                            
                            #l2 > l1
                            angularElem = (l2**2 - self.ml**2)/((2*l2+1)*(2*l2-1))
                            angularElem = sqrt(angularElem)
                            matrix[ind1][ind2] = radialInt * angularElem


                         
                            angularElem = ((l1 + 1)**2 - self.ml**2)/((2*l1+3)*(2*l1+1))
                            angularElem = sqrt(angularElem)
                            matrix[ind2][ind1] = radialInt * angularElem
        return matrix





