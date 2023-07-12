"""The splash submodule implements the SPLASH model :cite:p:`davis:2017a` for
calculating robust indices of radiation, evapotranspiration and plant-available
moisture.

It is based on the [SPLASH v1.0 implementation](https://doi.org/10.5281/zenodo.376293)
published with the original description but has been extensively refactored and
restructured. The main changes are:

* The implementation now provides a programmatic API to the main model components,
  allowing the functionality to be imported and used within Python workflows. The
  original code was implemented as command line scripts that required editing for
  particular use cases.

* The code has been updated to accept numpy arrays and to calculate estimates
  simulataneously across arrays. The original code predominantly expected scalar inputs
  and required users to script iteratation over sites.

* The majority of solar and evaporative fluxes do not vary are invariant given the
  initial data and so are now calculated when the model in instantiated. The previous
  code recalculated many of these values during iteration, which reduces the memory
  footprint but at a substantial runtime cost.

* The hard-coded constants in the original code can now be modified by users. In
  particular, the maximum soil moisture capacity (``kWn``) was fixed globally in SPLASH
  v1.0 at 150mm: this can now be set by the user and can vary between sites.
"""  # noqa: D205, D415
