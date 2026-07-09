r"""This module contains code and inputs used to generate a golden dataset for use in
testing the calculation of solar fluxes as implemented in the ``rsplash`` package. The
``pyrealm`` package does not attempt to calculate the other extensions included in
``rsplash`` but does provide the alternative calculation of solar fluxes from ground or
remotely sensed measurement of shortwave downwelling radiation, rather than from
sunshine fraction.

The `rsplash` package does not have accessors to output the internals of the solar
or evaporation calculations. The code in the example file was run using a modified
build of `rsplash` that was simply patched to include a call to the EVAP.display()
method within the loop in SPLASH.run_all() method and to add the calculation of sf to
that display output. The patch is saved as ``patch.diff``.


The file ``rsplash_Bourne_example.R`` contains R code to install the ``rsplash`` package
(https://github.com/dsval/rsplash/tree/master), export the Bourne forcing data for use
in pyrealm testing (``rsplash_Bourne_inputs.csv``) run the documented Bourne example and
save the predictions (``rsplash_Bourne_outputs.csv``). The example is modified
to set the site slope to zero to factor out slope and aspect influences on variables
that are not implemented in SPLASH v1.

The command line is then used to run the model and capture the output from display()
method and then parse the captured text into a data frame of the internal calculations
(``rsplash_Bourne_internal.csv``):

R --no-save < rsplash_Bourne_example.R > capture.out
python  parse_capture_to_csv.py

"""  # noqa: D205
