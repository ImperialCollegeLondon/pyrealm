---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Constrained Arrays

Many of the calculations in the `pyrealm` package only make sense for a limited
range of inputs and some, such as  `~pyrealm.pmodel.calc_density_h2o`, show
numerical instability outside of those ranges. The 
{mod}`~pyrealm.constrained_array` module provides mechanisms to apply and detect 
constraints.

## The {class}`~pyrealm.constrained_array.ConstrainedArray` class 

The {class}`~pyrealm.constrained_array.ConstrainedArray` class provides a 
mechanism to impose constraints on inputs and to make it easy to detect that
a constraint has already been imposed. The approach uses masked arrays 
({class}`~numpy.ma.core.MaskedArray`) from the `numpy` package to conceal out of 
range data without deleting it.

The class of an input can then be easily checked to see if a constraint has
already been imposed on the data.

```{code-cell} python
import numpy as np
from pyrealm.constrained_array import ConstrainedArray

# Some raw input data
vals = np.array([-15, 10, 20, 125])
print(vals)
```

```{code-cell} python
# Constrain the data to (0, 100)
vals_c = ConstrainedArray(vals, lower=0, upper=100)
print(vals_c)
```

```{code-cell} python
# Check that an input has been constrained
if isinstance(vals_c, ConstrainedArray):
    print('Input data has been constrained')
```

## The {func}`~pyrealm.constrained_array.constraint_factory` function

This utility function makes it easy to generate functions to impose a specific 
constraint. It takes a label and the constraint range and returns a function
that imposes that constraint and reports when data have been constrained using 
the label.


```{code-cell} python
from pyrealm.constrained_array import constraint_factory

# Create a constraint function for temperature in (0, 100) °C
temp_constraint = constraint_factory('temperature', 0, 100)

# Apply the constraint
vals_c = temp_constraint(vals)
```

```{code-cell} python
print(vals_c)
```

## Module documentation

```{eval-rst}
.. automodule:: pyrealm.constrained_array
    :members: ConstrainedArray, constraint_factory
```
