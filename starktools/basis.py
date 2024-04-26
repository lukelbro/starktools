import itertools
import numpy as np

class nlbasis:
    def __init__(self, nmin: int, nmax: int, ml=0):
        self.nmin = nmin
        self.nmax = nmax
        self.ml = ml
        self.basis_name = 'nlbasis'
    
    def __iter__(self):
            n = self.nmin
            l = np.abs(self.ml)
            while n <= self.nmax:
                if l < n:
                    yield (n, l)
                    l += 1
                else:
                    l = np.abs(self.ml)
                    n += 1
        
    def state_valid(self, state):
        try:
            n, l = state
        except:
            return False
        if n >= self.nmin and n <= self.nmax:
            if l >= np.abs(self.ml) and l < n:
                return True
        return False
    
    def state_valid_raise_exception(self, state):
        if not self.state_valid(state):
            raise ValueError('{state} is invalid in {basis_name} with parameters {basis_parameters}'.format(state=state, basis_name=self.basis_name, basis_parameters=self.basis_parameters()))
    
    def basis_parameters(self):
        return {'nmin': self.nmin, 'nmax': self.nmax, 'ml': self.ml}

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