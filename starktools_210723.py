import numpy as np
from math import exp, sqrt, ceil, log, floor
from functools import cached_property
from numba import jit
from constants import *

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

    @cached_property # array flattened to 2d
    def data(self):
        return self.make_array()

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
    
    def eigen(self):
        return np.linalg.eigh(self.data)[0]

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
        nn = np.arange(self.nmin, self.nmax+1, 1)
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
            frequency (float): frequency of side bands in atomic units
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

        # Make H0 array
        mm = self.gen_matrix_empty()
        for i in mm.keys(): 
            for j in mm[i].keys():
                for k in mm[i][j].keys():
                    mm[i][j][k][i][j][k] = -0.5 * (i - self._get_qd(i,j))**(-2) + k * frequency
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

                                angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
                                angularElem = sqrt(angularElem)

                                for k in mm[i][l1].keys():
                                    mm[i][l1][k][j][l2][k] = radialInt * angularElem
                            except KeyError:
                                pass

                            try:
                                test = mm[j][l2][0][i][l1][0]

                                angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
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

                                angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
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

                                angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
                                angularElem = sqrt(angularElem)
                                for k in mm[j][l2].keys():
                                    qi = k
                                    qj = qi + 1
                                    if (qj <= self.q):
                                        mm[i][l2][qi][j][l1][qj] = radialInt * angularElem
                                        mm[i][l2][qj][j][l1][qi] = radialInt * angularElem
                            except KeyError:
                                pass
        self.matrix = mm

class Tools:
    class numerov:
        def __init__(self, n: float, l: int, nmax = -1):
            self.__dict__['n'] = n
            self.__dict__['l'] = l
            self.__dict__['nmax'] = nmax


            self.r, self.y, self.start, self.end = Tools.numerov.integrate(n, l, nmax)
        
        @jit
        def integrate(n: float, l: int, nmax = -1, rCore=0.65):
            """
            Calculate radial integral for given state using numerov approach.
            n: Principle quantum number
            l: Azimuthal quantum number
            """

            if  l>=n:
                raise ValueError("Error: Azimuthal quantum number should be less than principle quantum number")
            if nmax == -1:
                nmax = n

            h = 0.01 # Step Size
            w = -0.5 *float(n)**-2 # Energy

            # Function for calculating g(x)
            g_func = lambda r: 2* r**2 * (-1.0/r - w) + (l + 0.5)**2
            
            # First starting point Y_{-1}
            rMax = 2 * nmax * (nmax + 15)
            if n == nmax: # Align wavefunctions using grid defined by max n in calculation
                i = 2 # Numerov approach starts at third point in itteration
                iStart = i - 2
                rStart1 = rMax
            else:
                i = int(floor(log(rMax / (2 * n * (n + 15))) / h))
                iStart = i
                rStart1 = rMax * exp(-i*h)
                i = i+2
            yStart1 = 1e-10

            # Secound starting point Y_{0}
            rStart2 = rStart1 * exp(-h)
            yStart2 = yStart1 * (1 + h*sqrt(g_func(rStart1)))

            # Define integration end points:
            rCore = 0.191**(1/3) # Core polarisability (specific to each atom species)
            rFugal = n**2 - n * sqrt(n**2 - l*(l+1)) # Inner turning point

            # Numerov Method: Need to figure out the number itterations that will be included in the integartion.
            ri = rStart1 * exp(-h*2)

            # Prepare arrays
            lengthMax = int(ceil(log(rMax/rCore)/h))
            r = np.zeros(lengthMax)
            y = np.zeros(lengthMax)
            r[i-2] = rStart1
            r[i-1] = rStart2
            y[i-2] = yStart1
            y[i-1] = yStart2

            while (ri > rCore):
                r[i] = ri

                A = y[i-2]*(g_func(r[i-2]) - 12/h**2)
                B = y[i-1]*(10*g_func(r[i-1]) + 24/h**2)
                C = 12/h**2 - g_func(r[i])

                y[i] = ((A + B)/C)

                if (ri < rFugal):
                    # Check for divergence
                    dy = abs((y[i] - y[i-1]) / y[i-1])
                    dr = (r[i]**(-l-1) - r[i-1]**(-l-1))/(r[i-1]**(-l-1))
                    if dy>dr:
                        break
                i += 1
                ri = ri * exp(-h)
            iEnd = i

            return r, y, iStart, iEnd


    def numerov_calc_matrix_element(wf1, wf2):
        """
        Calculate the radial component of the transition matrix element.
        Accepts two numrov integration data sets as tuples.
        """

        # Find range of points for which there are values for both wavefunctions
        iStart = max(wf1.start, wf2.start) # This does not work!!
        iEnd   = min(wf1.end, wf2.end)


        y1 = wf1.y[iStart:iEnd]
        y2 = wf2.y[iStart:iEnd]
        r = wf1.r[iStart:iEnd]

        # Calculate matrix element
        M = np.sum(y1 * y2 * r**3)

        # Normalise
        norm1 = (np.sum(wf1.y**2 * wf1.r**2))
        norm2 = (np.sum(wf2.y**2 * wf2.r**2))

        M = M/sqrt(norm1 * norm2)
        return M

class QuantumDefects:
    def __init__(self, defects = {}):
        self.__dict__['defects'] = defects

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

    def energy_level(self, n, l):
        return -0.5 * (n - self._get_qd(n,l))**(-2)


    def calc_matrix_element(self, n1, l1, n2, l2, nmax):
        """
        Calculates the dipole transition moment between two states, including the angular
        and radial componenets.
        """
        # Account for Quantum defects
        ns1 = n1 - self._get_qd(n1, l1)
        ns2 = n2 - self._get_qd(n2, l2)

        # Numerov integrals
        wf1 = Tools.numerov(ns1, l2, nmax)
        wf2 = Tools.numerov(ns2, l2, nmax)
        
        # Radial overlap
        radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)

        # Angular component
        m = 0
        angularElem = 0
        if l1>l2:
            angularElem = ((l1 + 1)**2 - m**2)/((2*l1+3)*(2*l1+1))
            angularElem = sqrt(angularElem)

        if l1<l2:
            angularElem = (l2**2 - m**2)/((2*l2+1)*(2*l2-1))
            angularElem = sqrt(angularElem)

        return radialInt * angularElem
