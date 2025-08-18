
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

# Utilities & environment checks

```{code-cell} python
import sys, numpy, scipy, matplotlib
print('Python', sys.version)
print('NumPy', numpy.__version__)
print('SciPy', scipy.__version__)
print('Matplotlib', matplotlib.__version__)
```

```{code-cell} python
# Ensure src is importable
import sys, os
sys.path.append('src')
import chpfc, plotting
print('Modules loaded.')
```
