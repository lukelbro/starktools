# Starktools

My personal library for calculating energy levels in Rydberg atoms.

```
pip install git+https://github.com/lukelbro/starktools
```

Rabi Frequency:
$$\Omega_{i,j} = \frac{\langle i| e\mathbf{r} \cdot \mathbf{E}_0 |j \rangle}{\hbar} $$


```
import starktools as st
electricDipoleMoment = qd.calc_matrix_element(55, 0, 55, 1, nmax = 80) * st.Constants.e * st.Constants.a_He

estrength = 0.1

rabi = electricDipoleMoment * estrength/st.Constants.hbar

```

```
def calc_transition(n1, l1, n2, l2):
    return st.Constants.c/(1/((qd.energy_level(n1, l1)*st.Constants.R_He  - qd.energy_level(n2, l2)*st.Constants.R_He)))*10**-9
```