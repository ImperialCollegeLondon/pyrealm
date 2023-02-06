---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The `rpmodel` implementation

The implementation and documentation of the {mod}`~pyrealm.pmodel` module is
very heavily based on Benjamin Stocker's R implementation of the P-model
({cite}`Stocker:2020dh`) in the [``rpmodel``](https://github.com/stineb/rpmodel)
package:

## Testing reference

The ``rpmodel`` package has been used to calculate the expected values that
have been used in testing {mod}`~pyrealm.pmodel`. This includes:

* the module docstrings, which are used to provide {mod}`doctest` tests of
  the basic operation of this implementation, and
* the module unit testing, provided using {mod}`pytest` in `test/test_pmodel.py`.

## Differences between `pyrealm` and `rpmodel`

The implementations differ in a number of ways:

1. The functions in ``rpmodel`` contains a large number of hard-coded
   parameters. These include both global constants and experimentally derived
   estimates for different processes. In the {mod}`~pyrealm.pmodel` module,
   these hard-coded parameters have been replaced within references to a global
   parameter dictionary ({const}`~pyrealm.params.PARAM`). This is used to share
   a set of constants and values across all functions. However, can also be
   altered, allowing users to explore model responses to underlying parameter
   variation (see [here](/params)).

   In some cases, ``rpmodel`` sets these constants in function arguments. These
   parameters have also been moved to {const}`~pyrealm.params.PModel_Params`
   to simplify function arguments.

1. The ``rpmodel`` package has suites of functions for calculating $J_{max}$
   limitation and optimal $\chi$. These have been combined into classes
   {class}`~pyrealm.pmodel.JmaxLimitation` and {class}`~pyrealm.pmodel.CalcOptimalChi`.
   This allows the common parameters and outputs of these functions to be standardised
   and the different methods are provided via a ``method`` argument to each class.

1. When simulating C4 plants, the ``rpmodel`` package, the ``rpmodel`` function
   enforces a separate $J_{max}$ method (``rpmodel:::calc_lue_vcmax_c4``). This
   is equivalent to the `simple` model with the $\ce{CO2}$ limitation factor
   $m_j=1.0$. Only this method can be used with C4 plants and hence it is not
   possible to simulate $J_{max}$ limitation for C4 plants. In the
   implementation in {mod}`~pyrealm.pmodel`, C4 plants are set to have no
   $\ce{CO2}$ limitation in {class}`~pyrealm.pmodel.CalcOptimalChi`, although the
   correct internal $\ce{CO2}$ partial pressure is calculated, and are then
   free to use whichever $J_{max}$ method is preferred in
   {class}`~pyrealm.pmodel.CalcLUEVcmax`.

1. The ``rpmodel`` function has a large number of arguments. This is partly
   because of some redundancy in identifying the use case. For example, using
   soil moisture stress can be inferred simply by providing inputs, rather then
   requiring the logical flag. However, the function also include steps that
   could be provided by the user. That means a user can put in all the variables
   in one function, but makes for a more confusing interface and is less
   flexible. The {class}`~pyrealm.pmodel.PModel` class reduces this to a simpler
   core of inputs and methods and functions reproduce the rest of the
   functionality in the P Model.

1. The ``rpmodel`` function has extended the implementation of the empirical
   soil moisture factor $\beta(\theta)$ from a simple factor on light use
   efficiency (LUE) to estimate the underlying values of $J_{max}$ and $V_{cmax}$.
   {class}`~pyrealm.pmodel.PModel` only applies $\beta(\theta)$ limitation
   factor to LUE and reports $J_{max}$ and $V_{cmax}$ estimates as if there was
   no limitation.
