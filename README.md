# Starktools

My personal library for calculating energy levels in Rydberg atoms.

```
pip install git+https://github.com/lukelbro/starktools
```

Rabi Frequency:
$$\Omega_{i,j} = \frac{\langle i| e\mathbf{r} \cdot \mathbf{E}_0 |j \rangle}{\hbar} $$


```
import starktools as st
qd = st.quantumdefects(st.helium_defects(S=1))  # triplet helium: 3S1, 3P2, 3D3, ...
electricDipoleMoment = qd.calc_matrix_element(55, 0, 55, 1, nmax = 80) * st.Constants.e * st.Constants.a_He

estrength = 0.1

rabi = electricDipoleMoment * estrength/st.Constants.hbar

```

Transition frequency in GHz:

```
qd.transition((56, 0), (55, 0))
```

```
1/((qd.transition((56, 0), (55, 0))*10**9/st.Constants.c) + 1/(786.8166e-9))*10**9
```