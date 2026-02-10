r"""This submodule contains benchmark outputs from the ``calculate_phi0.R`` script,
which isan experimental approach to calculating the :math:`\phi_0` parameter for
the P Modelwith modulation from climatic aridity and growing degree days and the
currenttemperature. The calculation is implemented in ``pyrealm`` as
:class:`~pyrealm.pmodel.quantum_yield.QuantumYieldSandoval`.

The files are:

* ``calculate_phi0.R``: The original implementation and parameterisation.
* ``create_test_inputs.R``: A script to run the original implementation with a range of
  inputs and save a file of test values.
* ``sandoval_kphio.csv``: The resulting test values.

"""  # noqa: D205
