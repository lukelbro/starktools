import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import starktools

m = starktools.MatrixH0(3, 5)
x = m.shape