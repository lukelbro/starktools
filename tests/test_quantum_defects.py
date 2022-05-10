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


def test_defect():
    qd = starktools.QuantumDefects(defects)
    assert qd._get_qd(55, 0) == approx(0.296669286287103, abs=1e-12) # confirmed using matlab
    assert qd._get_qd(55, 1) == approx(0.068354108675390) # confirmed using matlab


def test_energy_level():
    qd = starktools.QuantumDefects(defects)
    e55p = qd.energy_level(55, 0)
    assert e55p == -3.341738483799353e-04

def test_energy_level_si():
    qd = starktools.QuantumDefects(defects)
    R_He = starktools.Constants.R_He
    c  = starktools.Constants.c 
    h = starktools.Constants.h
    e55p = R_He * c * h * qd.energy_level(55, 0)
    assert e55p == approx(7.283563680292469e-22)

def test_calc_matrix_element():
    qd = starktools.QuantumDefects(defects)
    element = qd.calc_matrix_element(56, 0, 55, 1, 70)
    assert element == approx(1190, abs=10)
    element = qd.calc_matrix_element(55, 1, 56, 0, 70)
    assert element == approx(1190, abs=10)