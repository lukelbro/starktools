from math import sqrt

import numpy as np

from .constants import Constants
from .tools import Tools

# Helium Ritz coefficients [d0, d2, d4, ...], keyed by l, S, J.
HELIUM_FS = {
    0: {  # S states (l=0)
        0: {  # singlet
            0: [0.1397180648621, 0.02783573718, 0.0167922941, -0.001459031, 0.002922765]
        },
        1: {  # triplet
            1: [0.2966564877175, 0.03829666659, 0.007513112, -0.004547679, 0.00218014]
        }
    },
    1: {  # P states (l=1)
        0: {
            1: [-0.01214180360364, 0.007519080459, 0.0139778015, 0.004837312, 0.001228329]
        },
        1: {
            0: [0.0683280025127, -0.01864197524, -0.0123316557, -0.007951545, -0.00544810],
            1: [0.0683578576527, -0.01863046224, -0.0123304057, -0.007951245, -0.00545010],
            2: [0.0683602837923, -0.01862922821, -0.0123327551, -0.007952741, -0.0054519]
        }
    },
    2: {  # D states (l=2)
        0: {
            2: [0.00211337846449, -0.003090051058, 0.0000082722, -0.000309431, -0.00040114]
        },
        1: {
            1: [0.00288558028122, -0.006357601227, 0.0003366711, 0.000839416, 0.000379872],
            2: [0.00289094149325, -0.006357183630, 0.0003377711, 0.000839216, 0.000432375],
            3: [0.00289132882526, -0.006357704033, 0.0003367013, 0.000839518, 0.000381183]
        }
    },
    3: {  # F states (l=3)
        0: {
            3: [0.0004402942662, -0.00168944665, -0.000118320, 0.00032618]
        },
        1: {
            2: [0.0004448698922, -0.00173927524, 0.0001047676, 0.000033769],
            3: [0.0004485948328, -0.00172723230, 0.00015249, -0.000248683],
            4: [0.0004473792721, -0.00173921723, 0.0001047871, 0.000033164]
        }
    },
    4: {  # G states (l=4)
        0: {
            4: [0.00012473449079, -0.00079623012, -0.0001020553, -0.000013669]
        },
        1: {
            3: [0.0001257074312, -0.00079649819, -0.0000098081, -0.00001911],
            4: [0.0001287131610, -0.00079624615, -0.0000118966, -0.000014185],
            5: [0.0001271416711, -0.00079648417, -0.0000098575, -0.00001910]
        }
    },
    5: {  # H states (l=5)
        0: {
            5: [0.00004710089961, -0.000433227784, -0.0000081426]
        },
        1: {
            4: [0.00004779706743, -0.000433232255, -0.0000080716],
            5: [0.00004975761451, -0.000433227465, -0.0000081319],
            6: [0.00004872984645, -0.000433228157, -0.0000081016]
        }
    },
    6: {  # I states (l=6)
        0: {
            6: [0.00002186888117, -0.000261067322, -0.00000404867]
        },
        1: {
            5: [0.00002239075920, -0.000261068028, -0.00000404287],
            6: [0.00002376848314, -0.000261066218, -0.00000407658],
            7: [0.00002304760926, -0.000261067235, -0.0000040411]
        }
    }
}

_RTOL = 4 * np.finfo(float).eps


def helium_defects(S, J=None):
    """One Ritz row per l from HELIUM_FS, for calculations without spin.

    ``J`` maps l to the level used for that l; the default is the stretched
    level J = l + S (3S1, 3P2, 3D3, ... for S=1; 1S0, 1P1, ... for S=0).
    """
    J = J or (lambda l: l + S)
    return {l: HELIUM_FS[l][S][J(l)] for l in HELIUM_FS}


def ritz_defect(n, coeffs, rtol=_RTOL, maxiter=10):
    """Quantum defect from Ritz coefficients [d0, d2, d4, ...].

    Solves delta = sum_i d_i / (n - delta)^(2i) by fixed-point iteration from
    d0. A scalar is returned unchanged as an n-independent defect.
    """
    if np.ndim(coeffs) == 0:
        return float(coeffs)
    delta = coeffs[0]
    for _ in range(maxiter):
        new = sum(d / (n - delta) ** (2 * i) for i, d in enumerate(coeffs))
        if abs(new - delta) <= rtol * abs(new):
            return new
        delta = new
    raise ValueError(f"Ritz expansion did not converge for n={n}, coefficients {coeffs}")


def lookup_defect(defects, n, l, s=None, j=None):
    """Quantum defect from a table keyed by l, or by l, S, J when s and j are given.

    l beyond the table is hydrogenic (zero defect); an S or J missing for a
    tabulated l raises KeyError.
    """
    if l not in defects:
        return 0.0
    row = defects[l]
    if s is not None:
        row = row[s][j]
    return ritz_defect(n, row)


class quantumdefects:
    """Quantum defects and energies from a table keyed by l, or by l, S, J."""

    def __init__(self, defects=None):
        self.defects = {} if defects is None else defects

    def quantum_defect(self, n, l, s=None, j=None):
        return lookup_defect(self.defects, n, l, s, j)

    def energy_level(self, n, l, s=None, j=None):
        """Energy in Hartree."""
        return -0.5 * (n - self.quantum_defect(n, l, s, j)) ** (-2)

    def transition(self, state1, state2):
        """E(state1) - E(state2) in GHz; states are (n, l) or (n, l, s, j)."""
        dE = self.energy_level(*state1) - self.energy_level(*state2)
        return dE * Constants.E_He / Constants.h * 1e-9

    def calc_matrix_element(self, n1, l1, n2, l2, nmax, ml=0):
        """
        Calculates the dipole transition moment between two states, including the angular
        and radial componenets.
        """
        if np.abs(l1 - l2) != 1:
            return 0
        # Account for Quantum defects
        ns1 = n1 - self.quantum_defect(n1, l1)
        ns2 = n2 - self.quantum_defect(n2, l2)

        # Numerov integrals
        wf1 = Tools.numerov(ns1, l1, nmax)
        wf2 = Tools.numerov(ns2, l2, nmax)

        # Radial overlap
        radialInt = Tools.numerov_calc_matrix_element(wf1, wf2)

        # Angular component
        angularElem = 0
        if l1>l2:
            angularElem = ((l2 + 1)**2 - ml**2)/((2*l2+3)*(2*l2+1))
            angularElem = sqrt(angularElem)

        if l1<l2:
            angularElem = (l2**2 - ml**2)/((2*l2+1)*(2*l2-1))
            angularElem = sqrt(angularElem)

        return radialInt * angularElem
