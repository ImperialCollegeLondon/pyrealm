r"""This module contains code and inputs used to generate a golden dataset for use in
testing the calculation of solar fluxes as implemented in the ``rsplash`` package. The
``pyrealm`` package does not attempt to calculate the other extensions included in
``rsplash`` but does provide the alternative calculation of solar fluxes from ground or
remotely sensed measurement of shortwave downwelling radiation, rather than from
sunshine fraction.

The file ``rsplash_Bourne_example.R`` contains R code to install the ``rsplash`` package
(https://github.com/dsval/rsplash/tree/master), run the documented Bourne example and
save the predictions to the file ``rsplash_Bourne_outputs.csv`` .

The `rsplash` package does not have accessors to output the internals of the solar
or evaporation calculations. The code in the example file was run using a modified
build of `rsplash` that was simply patched to include a call to the EVAP.display()
method within the loop in SPLASH.run_all() method:

diff --git a/src/SPLASH.cpp b/src/SPLASH.cpp
index 6c99c33..8c318cd 100644
--- a/src/SPLASH.cpp
+++ b/src/SPLASH.cpp
@@ -1906,6 +1906,8 @@ List SPLASH::run_all(vector<int> &doys, vector<int> &yrs, vector<double> &sw_in,
         cond_vec[i] = dvap.cond;
         netr_vec[i] = dvap.rn_d/1e6;
         nds_vec[i] = dsoil.nd;  // # codespell:ignore
+
+        evap.display();
     }

"""  # noqa: D205, E501
