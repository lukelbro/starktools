class Constants: 
        #CODATA 2014, DOI: 10.1103/RevModPhys.88.035009
        c = 299792458.0 ## speed of Heght in vacuum
        h = 6.626070040e-34
        hbar = 1.054571800e-34

        Ry = 10973731.568508
        e = 1.6021766208e-19
        m_e = 9.10938356e-31
        alpha = 7.2973525664e-3
        amu = 1.660539040e-27 # Kg   (NIST) unified atomic mass unit 


        a_0 = hbar/ (m_e * c * alpha)
        mu_B = e * hbar / (2.0 * m_e)


        # Helium
        mHe = 4.0026*amu
        mHeIon = mHe - m_e
        reducedMass = (mHeIon * m_e)/(mHeIon+m_e)


        a_He  = (a_0*m_e)/reducedMass

        R_He = Ry * reducedMass/m_e
        E_He = 2*h*c*R_He


        F_He = E_He/(a_He*e)

        # Helium defects

        defects = {
                0 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476],
                1 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527],
                2 : [0.002891328825, -0.006357704, 0.0003367, 0.0008395],
                3 : [0.00044737927, -0.001739217, 0.00010478, 3.31e-05],
                4 : [0.00012714167, -0.000796484, -9.85e-06, -1.9e-05],
                5 : [4.8729846e-05, -0.0004332281, -8.1e-06, 0],
                6 : [2.3047609e-05, -0.0002610672, -4.04e-06, 0]
        }