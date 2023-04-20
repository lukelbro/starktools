
from math import exp, sqrt, ceil, log, floor
import numpy as np

class Tools:
    class numerov:
        def __init__(self, n: float, l: int, nmax = -1):
            self.__dict__['n'] = n
            self.__dict__['l'] = l
            self.__dict__['nmax'] = nmax

            self.r, self.y, self.start, self.end = Tools.numerov.integrate(n, l, nmax)

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
                i = 2 # Numerov approach starts at third point in iteration
                iStart = i - 2
                rStart1 = rMax
            else:
                i = int(floor(log(rMax / (2 * n * (n + 15))) / h)) # this step results in the matrix not being perfectly hermitian!
                                                                    # as the grid is discrete so the choice in grid position results
                                                                    # in an approximation!
                iStart = i
                rStart1 = rMax * exp(-i*h)
                i = i+2
            yStart1 = 1e-10

            # Second starting point Y_{0}
            rStart2 = rStart1 * exp(-h)
            yStart2 = yStart1 * (1 + h*sqrt(g_func(rStart1)))

            # Define integration end points:
            rCore = 0.191**(1/3) # Core polarisability (specific to each atom species)
            rFugal = n**2 - n * sqrt(n**2 - l*(l+1)) # Inner turning point

            # Numerov Method: Need to figure out the number iterations that will be included in the integration.
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
        imin = max(wf1.start, wf2.start)
        imax = min(wf1.end, wf2.end)

        y1 = wf1.y[imin:imax]
        y2 = wf2.y[imin:imax]
        r =  wf1.r[imin:imax]
    

        # Calculate matrix element
        M = np.sum(y1 * y2 * r**3)

        # Normalise
        norm1 = (np.sum(wf1.y**2 * wf1.r**2))
        norm2 = (np.sum(wf2.y**2 * wf2.r**2))

        M = M/sqrt(norm1 * norm2)
        return M