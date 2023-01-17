# Matplotlib Inline Back-end for IPython and Jupyter

This package provides support for matplotlib to display figures directly inline in the Jupyter notebook and related clients, as shown below.

## Installation

With conda:

```bash
conda install -c conda-forge matplotlib-inline
```

With pip:

```bash
pip install matplotlib-inline
```

## Usage

Note that in current versions of JupyterLab and Jupyter Notebook, the explicit use of the `%matplotlib inline` directive is not needed anymore, though other third-party clients may still require it.

This will produce a figure immediately below:

```python
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3*np.pi, 500)
plt.plot(x, np.sin(x**2))
plt.title('A simple chirp');
```

## License

Licensed under the terms of the BSD 3-Clause License, by the IPython Development Team (see `LICENSE` file).
