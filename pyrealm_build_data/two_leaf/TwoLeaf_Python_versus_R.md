---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: pyrealm_python3
  language: python
  name: pyrealm_python3
---

# Compare R to Python draft implementations

The code below loads Keith's outputs from the R draft implementation and David's
outputs from the parallel Python draft and aligns them.

```{code-cell} ipython3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load the R outputs
r_outputs = pd.read_csv('two_leaf_R_implementation_outputs.csv')
r_outputs['time'] = pd.to_datetime(r_outputs['step'])

# Load the Python outputs
py_outputs = pd.read_csv('two_leaf_python_implementation_outputs.csv')
py_outputs['time'] = pd.to_datetime(py_outputs['time'], utc=True)

# Reduce the Python outputs to observations present in the R outputs:
# - August only
# - but _also_ some timestamps with low quality dropped in R.
py_outputs = py_outputs[py_outputs['time'].isin(r_outputs['time'])]
```

Now create plots to show correlation plots between the two implementations and overlying
short time series for the two implementations

```{code-cell} ipython3
# Setup plots: Split variables into page size chunks of variables, which splits
# the plots across pages more cleanly and set the time series length.
vars_per_page = 4
N_time = 48 * 3
vars_to_plot = py_outputs.columns[1:].to_numpy()
n_chunks = np.ceil(len(vars_to_plot) / vars_per_page)
vars_to_plot = np.array_split(vars_to_plot, n_chunks)

# Loop over variables sets
for var_set in vars_to_plot:

    # Create a 6 by 2 plot for these variables
    fig, axes = plt.subplots(ncols=2, nrows=vars_per_page, figsize=(7, 12))

    # Loop over variables
    for var, (ax1, ax2) in zip(var_set, axes):
        
        # Correlation plot
        ax1.scatter(r_outputs[var], py_outputs[var])
        all_vals = np.concatenate([r_outputs[var], py_outputs[var]])
        minmax= [np.nanmin(all_vals), np.nanmax(all_vals)]
        ax1.plot(minmax, minmax, color='r')
        ax1.set_ylabel(var + ' python')
        ax1.set_xlabel(var + ' r')
        
        # Short time series
        ax2.plot(r_outputs['time'][:N_time], r_outputs[var][:N_time], label='R')
        ax2.plot(py_outputs['time'][:N_time], py_outputs[var][:N_time], label='Py')
        ax2.set_xlabel("Time")
        ax2.set_ylabel(var);
        ax2.legend(frameon=False, ncols=2)

    plt.tight_layout()
```

```{code-cell} ipython3

```
