
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

# CH–PFC model and parameters

We follow the coupled free energy described in Eq. (A2), with the anisotropic, composition-dependent Laplacian ∇^2_c defined via the transformation matrix **A**(c) [Eq. (A3–A4) in the Appendix]. Lattice parameters for FePO$_4$ and LiFePO$_4$ used here are taken from Table I. citeturn1search1

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Make src importable
import os
if 'src' not in os.listdir():
    # when running in Colab after upload, adjust path if needed
    pass
sys.path.append('src')

from chpfc import Params, make_grid, init_fields, pfc_relax
from plotting import plot_fields

p = Params(nx=192, ny=192, Lx=192, Ly=192, dx=1.0, r_electrode=70,
           kappa=1.0, RT=1.0, Om=2.0, xi=1.0, r_inside=0.2, r_outside=-0.2)
print('α_FP, β_FP, α_LFP, β_LFP =', p.alpha_fp, p.beta_fp, p.alpha_lfp, p.beta_lfp)
```

We initialize a circular electrode (crystalline) embedded in an amorphous Li reservoir by setting the control field $r(\mathbf{x})$ to $+0.2$ inside and $-0.2$ outside, matching the ordered vs. disordered behavior of the PFC field as described in Sec. II and Fig. 2. citeturn1search1

```{code-cell} python
c, psi, r, mask = init_fields(p, seed=42)
fig, axs = plot_fields(c, psi, mask, title_prefix='Initial')
plt.show()
```

We relax the PFC peak-density field $\psi$ to equilibrium for the given composition and $r(\mathbf{x})$, assuming fast elastic relaxation compared to diffusion (Eq. (3) with $\delta F/\delta\psipprox 0$), as done in the paper. citeturn1search1

```{code-cell} python
psi = pfc_relax(psi, c, r, p, n_steps=200, dt=0.05)
fig, axs = plot_fields(c, psi, mask, title_prefix='After PFC relax')
plt.show()
```
