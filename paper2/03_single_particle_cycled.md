
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

# Single-particle lithiation (cycled electrode)

We lithiate a polycrystalline FePO$_4$ particle by holding the Li reservoir composition fixed at $c=1$ and evolving the electrode composition via the Cahn–Hilliard equation (boundary-condition-driven diffusion), as described in Sec. III.B and Fig. 3. citeturn1search1

```{code-cell} python
import numpy as np, matplotlib.pyplot as plt, sys, os
sys.path.append('src')
from chpfc import Params, init_fields, pfc_relax, ch_step, extract_peak_coords
from plotting import plot_fields

p = Params(nx=192, ny=192, r_electrode=70)
c, psi, r, mask = init_fields(p, seed=0)
psi = pfc_relax(psi, c, r, p, n_steps=200, dt=0.05)

snapshots = []
T_total = 100
for t in range(T_total):
    # enforce reservoir boundary c=1 outside electrode
    c[~mask] = 1.0
    c = ch_step(c, psi, p, dt=5e-3, n_sub=5)
    # fast PFC relaxation step every few diffusion steps
    if t % 5 == 0:
        psi = pfc_relax(psi, c, r, p, n_steps=20, dt=0.05)
    if t in [0, 30, 60, 99]:
        snapshots.append((t, c.copy(), psi.copy()))

len(snapshots)
```

```{code-cell} python
for t, cc, pp in snapshots:
    fig, axs = plot_fields(cc, pp, mask, title_prefix=f't={t}')
    plt.show()
```

We now compute *peak-marker* images and a coarse-grained distortion map $\delta(\mathbf{x})$ based on Voronoi centroids, analogous to the paper (Sec. III.A–B). citeturn1search1

```{code-cell} python
from scipy.ndimage import gaussian_filter

peaks_over_time = []
for t, cc, pp in snapshots:
    coords = extract_peak_coords(pp, threshold=0.02, min_distance=2)
    peaks_over_time.append((t, coords))

# Distortion relative to first snapshot: use raw peak coordinates as proxies for Voronoi centroids
ref_t, ref_coords = peaks_over_time[0]
def to_xy(coords):
    return np.array([[float(j), float(i)] for i,j in coords])
ref_xy = to_xy(ref_coords)

for (t, coords) in peaks_over_time:
    xy = to_xy(coords)
    # match lengths by truncation for simplicity
    m = min(len(xy), len(ref_xy))
    if m == 0:
        continue
    d = np.linalg.norm(xy[:m] - ref_xy[:m], axis=1)
    print(f't={t:3d}: mean |Δ| ≈ {d.mean():.3f} (pixels)')
```
