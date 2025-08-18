
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Noncycled electrode (curvature-driven grain motion)

We evolve the same initial particle **without** lithiation (homogeneous $c=0$), relaxing only the PFC field over an equivalent total time, to compare with Fig. 4. citeturn1search1

```{code-cell} python
import numpy as np, matplotlib.pyplot as plt, sys
sys.path.append('src')
from chpfc import Params, init_fields, pfc_relax
from plotting import plot_fields

p = Params(nx=192, ny=192, r_electrode=70)
c, psi, r, mask = init_fields(p, seed=1)
# make homogeneous c=0 everywhere in the domain for the noncycled case
c[:] = 0.0

snapshots_nc = []
psi0 = psi.copy()
psi = pfc_relax(psi, c, r, p, n_steps=500, dt=0.05)
snapshots_nc.append((0, c.copy(), psi0))
snapshots_nc.append((1, c.copy(), psi.copy()))

for t, cc, pp in snapshots_nc:
    fig, axs = plot_fields(cc, pp, mask, title_prefix=f'noncycled t={t}')
    plt.show()
```
