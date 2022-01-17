from .context import starktools
import numpy as np
from pytest import approx, raises

def test_constants_h():
    assert starktools.Constants.h ==  approx(6.62606957e-34) # slight difference with matlab version

def test_constants_c():
    assert starktools.Constants.c == 299792458
    
def test_constants_hbar():
    assert starktools.Constants.hbar == approx(6.62606957e-34/(2*np.pi))

def test_constant_Ry():
    assert starktools.Constants.Ry == approx(109737.31568539*100)

def test_constant_R_He():
    assert starktools.Constants.R_He == approx(1.097222657637132e+05*100)
    
def test_constant_E_He():
    E = starktools.Constants.E_He/2
    assert  E == approx(2.179573211848553e-18)

    # assert starktools.Constants.e == 1.6021766208e-19
    # assert starktools.Constants.a_0 == 0.52917721092e-10


    #     Ry_inf = 109737.31568539; % 1/cm (NIST)
    # me = 9.10938291e-31;       % Kg   (NIST)
    # mp = 1.672621777e-27;      % Kg   (NIST)
    # mn = 1.674927351e-27;      % Kg   (NIST)
    # amu = 1.660538921e-27;     % Kg   (NIST) unified atomic mass unit 
    # B0 = 2.35e5;              % T
    # F0 = 5.14e9;              % V/cm
    # debye = 3.33564e-30;      % Cm   (NIST) electric dipole moment

    # e = 1.602176565e-19;       % C    (NIST)
    # h = 6.62606957e-34;        % Js   (NIST)
    # hbar = h / (2 * pi);
    # c = 299792458;            % m/s  (NIST)
    # a0 = 0.52917721092e-10;           % m    (Gallagher book)
    # alpha = 7.2973525698e-3;