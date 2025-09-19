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

Most of the functionality in `pyrealm` is designed to work with data arrays that can
have multiple dimensions. The input data can be single scalar values - representing a
point estimate - or multi-dimensional inputs - such as a time-series on a spatial grid.
When inputs have multiple dimensions, these can be thought of as **axes**: a 3D array
might have a time axis, an X axis and a Y axis. The **shape** of an array is then just
the number of observations along each axis

If all of the inputs vary over all of the axes, then the input data all will need to
have the same shape. However if one of the inputs is constant along one of the axes then
that data can be repeated across the other axes. As an example, a function might need
the following two inputs:

* The temperature, which is a 5 x 5 spatial grid on a time series with five
  observations and therefore has the shape `(5, 5, 5)`.
* The elevation, which uses the same 5 x 5 spatial grid but is constant with time and so
  has the shape `(5, 5)`

Any calculation with these inputs has a problem: which two axes in the temperature data
does the elevation map onto? In earlier versions of `pyrealm` (<=`1.0.0`), users were
required to manually make all inputs the same shape by repeating data along axes.
However, from `pyrealm 2.0.0`, users just need to make sure that inputs have compatible
shapes, as described below.

Note that **scalar inputs** (a single value) are a special case and are automatically
assumed to be constant across all the other inputs

## Array shapes

In `pyrealm`, we make use of the [array broadcasting
rules](https://numpy.org/doc/stable/user/basics.broadcasting.html) of the `numpy`
package. All array inputs to a `pyrealm` function then must be **mutually
broadcastable**. It is worth reading that link but, in summary, the arrays must:

* have the same number of dimensions, and
* have the same shape in each dimension or only a single observation in that dimension.

Some examples:

❌ Two arrays with shapes `(3, 5)` and `(5,)` are not compatible because the
first array has two dimensions and the second only has a single dimension.

❌ Two arrays with shapes `(3, 5)` and `(5, 3)` are not compatible. They are both
two-dimensional but have different sizes along the two axes.

✅ Two arrays with shapes `(3, 5)` and `(3, 5)` are compatible. They have the same
number of dimensions and the same shape.

✅ Two arrays with shapes `(3, 5)` and `(1, 5)` are also compatible. They are both 2
dimensional and the first axis in the second array has a single observation and so can
be broadcast across the three observations in the first array.

✅ Using the earlier example with temperature and precipitation, if the order of the
dimensions on the temperature array is `(x, y, time)`, then the shape of
elevation should be set to be `(5, 5, 1)`. That final singleton dimension then give an
unambiguous relationship between the two inputs.

```{caution}
If you are used to using [the R language](https://www.r-project.org/), note that `numpy`
does not "recycle" values in the same way. R generally recycles shorter dimensions of
any length to match longer dimensions but `numpy` does not do this: values are only
"recycled" by broadcasting when there is a **single** observation on a dimension. If you
wanted to repeat pairs of values along a dimension, then you will need to do so manually.
```

As an example of how to do this in practice, in the following example most of the
arguments to {class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` are
2-dimensional in time and position. But the pressure 1-dimensional --- constant in time.
Therefore, the time axis is added before passing it to
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

# Show the shapes for comparison
print(temp.shape)
print(patm.shape)

# Call PModelEnvironment with all the inputs
env = PModelEnvironment(tc=temp, co2=co2, patm=patm, vpd=vpd, fapar=fapar, ppfd=ppfd)
```
