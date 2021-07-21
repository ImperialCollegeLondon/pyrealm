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
range of inputs and some, such as  {func}`~pyrealm.pmodel.calc_density_h2o`, show
numerical instability outside of those ranges. The 
{mod}`~pyrealm.constrained_array` module provides mechanisms to apply and detect 
constraints.

## The {class}`~pyrealm.constrained_array.ConstrainedArray` class 

The {class}`~pyrealm.constrained_array.ConstrainedArray` class imposes 
constraints on an array or numeric input and records that a constraint 
has been imposed. The approach uses masked arrays ({class}`~numpy.ma.core.MaskedArray`) 
from the `numpy` package to conceal out of range data without deleting it.

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

The {attr}`interval_type` parameter to {class}`~pyrealm.constrained_array.ConstrainedArray`
can be used to set with the constraint interval is open, closed or half-closed.

```{code-cell} python
# Constrain the data to [0, 125]
vals_c = ConstrainedArray(vals, lower=0, upper=125, interval_type='[]')
print(vals_c)
```

```{code-cell} python
# Constrain the data to (0, 125)
vals_c = ConstrainedArray(vals, lower=0, upper=125, interval_type='()')
print(vals_c)
```

## The {class}`~pyrealm.constrained_array.ConstraintFactory` class

This is a utility class which can be used to create ConstraintFactory instances
with a particular set of constraints and label. The instances are callable and
so can be used easily to impose the same set of constraints on different inputs. 

{class}`~pyrealm.constrained_array.ConstraintFactory` instances can take 
existing {class}`~pyrealm.constrained_array.ConstrainedArray` instances as an
input. The class will check that the input constraints match the factory 
constraints.     


```{code-cell} python
from pyrealm.constrained_array import ConstraintFactory

# Create a constraint function for temperature in (0, 100) °C
temp_constraint = ConstraintFactory(0, 100, label='temperature (°C)')

# The resulting class instance has a human readable representation
temp_constraint
```

```{code-cell} python
# Apply the constraint - showing the warning about masking.
vals_c = temp_constraint(vals)
```

```{code-cell} python
print(vals_c)
```

## Module documentation

```{eval-rst}
.. automodule:: pyrealm.constrained_array
    :members: ConstrainedArray, ConstraintFactory
```
