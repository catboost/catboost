# Matplotlib Inline Back-end for IPython and Jupyter

## Installation

With conda:

```bash
conda install -c conda-forge notebook matplotlib
```

With pip:

```bash
pip install notebook matplotlib
```

## Usage

This package is included in IPython and can be used in a Jupyter Notebook:

```python
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3*np.pi, 500)
plt.plot(x, np.sin(x**2))
plt.title('A simple chirp');
```
