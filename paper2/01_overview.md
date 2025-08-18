
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

# Reproducing *Phase-field model for diffusion-induced grain boundary migration* (Balakrishna et al., 2019)

This Jupyter Book is a runnable, teaching-oriented reproduction of key results from:

> **A. Renuka Balakrishna, Yet-Ming Chiang, W. Craig Carter** (2019). *Phase-field model for diffusion-induced grain boundary migration: An application to battery electrodes*, **Physical Review Materials 3, 065404**.  
> DOI: 10.1103/PhysRevMaterials.3.065404  
> We base equations, parameters and figures on the paper and its Appendix.  

:::{note}
We implement a minimal CH–PFC solver in pure NumPy with periodic finite differences. To keep run-times practical in Colab, we adopt the same separation of time scales as the paper (fast elastic/PFC relaxation compared to diffusion) and drop the small coupling term in Eq. (A5) when computing the Cahn–Hilliard chemical potential (as discussed in the Appendix). This yields qualitatively comparable behavior while preserving the central mechanisms reported.  
:::

## How to use this book (Colab-friendly)

- Open each chapter on Colab via the badge below and run cells sequentially.  
- Start with the **Model & Parameters**, then the **Single-particle (cycled)** and **Noncycled** notebooks.  
- The **Grain growth stats** chapter batches multiple random microstructures to reproduce the statistical comparison of Fig. 5 and Table III.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/chpfc_jupyter_book)

## Citation
If you use this reproduction, please cite the original paper above.
