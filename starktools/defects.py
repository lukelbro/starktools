from .tools import Tools
from math import sqrt
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
        return - (n - self._get_qd(n,l))**(-2)


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