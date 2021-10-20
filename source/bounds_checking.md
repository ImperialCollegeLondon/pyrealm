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

# Bound checking

Many of the calculations in the `pyrealm` package only make sense for a limited
range of inputs and some, such as  {func}`~pyrealm.pmodel.calc_density_h2o`, show
numerical instability outside of those ranges. The 
{mod}`~pyrealm.bounds_checking` module provides mechanisms to apply and detect 
constraints.

## The {func}`~pyrealm.bounds_checking.bounds_checker` function 

The {func}`~pyrealm.bounds_checking.bounds_checker` function is a simple
pass through function, often used in initialising class instances. It
simply detects whether provided values fall within a simple lower and upper
bound and issues a warning when they do not

## The {func}`~pyrealm.bounds_checking.input_mask` function 

The {func}`~pyrealm.bounds_checking.input_mask` function does much the same thing
but modifies the input values to replace out of bounds values with `np.nan`.


## Module documentation

```{eval-rst}
.. automodule:: pyrealm.bounds_checker
    :members: bounds_checker, input_mask
```
