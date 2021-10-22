from .context import starktools
import numpy as np
from pytest import approx, raises

defects = {
    0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
    1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
    2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
    3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
    4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
    5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
    6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
}

def test_numerov_method():
    nmin = 50
    nmax = 53
    qd = starktools.QuantumDefects(defects)

    wf1 = starktools.Tools.numerov(19, 0 ,19)
    assert wf1.y[100] == 0.00013208511620903889


def test_numerov():
    wf1  = starktools.Tools.numerov(19, 0, 19)
    wf2 = starktools.Tools.numerov(13, 1, 19)
    d = starktools.Tools.numerov_calc_matrix_element(wf1, wf2)

    assert d == -4.008890618586264 # confirmed using matlab