---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# Using arrays in `pyrealm`

Many functions in `pyrealm` accept array inputs. Unless stated in their descriptions
these must be follow the rules below.

Array inputs can be either NumPy arrays or scalar values. Arrays of shape `(1,)` are
also treated as scalar values. Multidimensional arrays are allowed, but they must have
compatible shapes.

## Array shapes

NumPy array inputs must be mutually broadcastable and have the same number of
dimensions. For example, two arrays with shapes `(3, 5)` and `(1, 5)` are acceptable
because the second array is assumed to be constant along the first dimension. However
arrays with shapes `(3, 5)` and `(5,)` are incompatible because the number of dimensions
is not the same.

In the following example most of the arguments to
{class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` are 2-dimensional in time
and position. But the pressure 1-dimensional --- constant in time. Therefore, the time
axis is added before passing it to
{class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`.

```{code-cell} ipython3
import numpy as np
from pyrealm.pmodel.pmodel_environment import PModelEnvironment

n_time = 10
n_position = 100

# 2D arrays in time and position (a single dummy value is used)
temp = np.full((n_time, n_position), 20)
co2 = np.full((n_time, n_position), 400)
vpd = np.full((n_time, n_position), 1000)
fapar = np.full((n_time, n_position), 1)
ppfd = np.full((n_time, n_position), 800)

# 1D array in position - constant in time
patm = np.full(n_position, 101325)

# Declare a new axis to make it 2D with shape (1, n_position)
patm = patm[np.newaxis, :]

# Call PModelEnvironment with all the inputs
env = PModelEnvironment(tc=temp, co2=co2, patm=patm, vpd=vpd, fapar=fapar, ppfd=ppfd)
```
