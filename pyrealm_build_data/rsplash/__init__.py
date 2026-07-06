r"""This module contains code and inputs used to generate a golden dataset for use in
testing the calculation of solar fluxes as implemented in the ``rsplash`` package. The
``pyrealm`` package does not attempt to calculate the other extensions included in
``rsplash`` but does provide the alternative calculation of solar fluxes from ground or
remotely sensed measurement of shortwave downwelling radiation, rather than from
sunshine fraction.

The file ``rsplash_Bourne_example.R`` contains R code to install the ``rsplash`` package
(https://github.com/dsval/rsplash/tree/master), run the documented Bourne example and
save the predictions to the file ``rsplash_Bourne_outputs.csv`` . Only one of these
(``netr``, net radiation) is used as a test case in ``pyrealm``.
"""  # noqa: D205
