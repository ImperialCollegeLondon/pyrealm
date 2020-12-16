# The `rpmodel` implementation


The implementation and documentation in {mod}`pyrealm.pmodel` is very heavily
based on Benjamin Stocker's R implementation of the P-model
\cite{Stocker:2020dh}. The `rpmodel` package is developed here:

[https://github.com/stineb/rpmodel]([https://github.com/stineb/rpmodel])


## Testing reference

The ``rpmodel`` package has been used to calculate the expected values that 
have been used in testing `pyrealm.pmodel`. This includes:

* the module docstrings, which are used to provide {mod}`doctest` tests of
the basic operation of this implementation, and
* the module unit testing, provided using {mod}`pytest` in `test/test_pmodel.py`.

## Differences between `pyrealm` and `rpmodel`

The implementations differ in a number of ways:

1. The functions in `rpmodel` contains a large number of hard-coded parameters.
   These include both global constants and experimentally derived estimates for
   different processes. In the {mod}`pyrealm.pmodel` module, these hard-coded
   parameters have been replaced within references to a global parameter list
   ({const}`pyrealm.params.PARAM`). This is used to provide values to all
   functions and is also editable, to allow users to explore model responses to
   underlying parameter variation.

2. Similarly, some functions in {mod}`pyrealm:pmodel` have shorter argument
   lists, either because the parameter in question is now defined in
   {const}`pmodel.PARAM` or because code reorganisation for Python has allowed
   for a simpler interface.

3. The `rpmodel` package has suites of functions for calculating $J_{max}$
   limitation and optimal $\chi$. These have been combined into classes
   {class}`CalcLUEVcmax` and {class}`CalcOptimalChi`. This allows the common
   parameters and outputs of these functions to be standardised and the
   different methods are provided via a ``method`` argument to each class.

4. When simulating C4 plants, the `rpmodel` package, the {func}`pmodel` function
   imposes the use of a separate $J_{max}$ method
   (`rpmodel:::calc_lue_vcmax_c4`). This is equivalent to the simple model given
   no $\ce{CO2}$ limitation. In the {mod}`pyrealm.pmodel` implementation, C4
   plants are set to have no $\ce{CO2}$ limitation  and then will use whichever
   $J_{max}$ method is selected.


