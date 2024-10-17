"""The `t_model` submodule provides reference data for testing the implementation of the
T model :cite:p:`Li:2014bc`. The file ``t_model.r`` contains the original implementation
in R.  The ``rtmodel_test_outputs.r`` contains a slightly modified version of the
function that makes it easier to output test values and then runs the function for the
following scenarios:

* A 100 year sequence of plant growth for each of three plant functional type (PFT)
  definitions (``default``, ``alt_one`` and ``alt_two``). The parameterisations for the
  three PFTs are in the file ``pft_definitions.csv`` and the resulting time series for
  each PFT is written to ``rtmodel_output_xxx.csv``.

* Single year predictions across a range of initial diameter at breast height values for
  each of the three PFTs. These are saved as ``rtmodel_unit_testing.csv`` and are used
  for simple validation of the main scaling functions.

To generate the predicted outputs again requires an R installation

.. code:: sh

    Rscript rtmodel_test_outputs.r

"""  # noqa: D205, D415
