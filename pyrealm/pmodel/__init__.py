"""The :mod:`~pyrealm.pmodel` module includes four submodules:

* The :mod:`~pyrealm.pmodel.pmodel` submodule provides classes implementing the core
  calculations of the P  Model.
* The :mod:`~pyrealm.pmodel.functions` submodule provides key standalone functions for
  those calculations.
* The :mod:`~pyrealm.pmodel.isotopes` submodule provides a class to estimate isotopic
  discrimination within the P Model.
* The :mod:`~pyrealm.pmodel.competition` submodule provides a competition model for the
  C3 and C4 photosynthetic pathways.

Note that the documentation of functions and methods includes two lists of parameters:

Parameters
  These are the arguments specific to the class, method or function signature.

Constants
  These are shared parameters of the PModel , which are taken from the
  :class:`~pyrealm.constants.pmodel_const.PModelConst` dataclass. These can be changed
  by the user but are typically used to configure an entire analysis rather than a
  single function.
"""  # noqa: D210, D415

# This __init__ file imports the following members from submodules in order to
# flatten the namespace for the main public components and setup.cfg applies
# # noqa: F401 to the whole file.

from pyrealm.pmodel.competition import (
    C3C4Competition,
    calculate_tree_proportion,
    convert_gpp_advantage_to_c4_fraction,
)
from pyrealm.pmodel.fast_slow_scaler import FastSlowScaler
from pyrealm.pmodel.functions import (
    calc_co2_to_ca,
    calc_density_h2o,
    calc_ftemp_arrh,
    calc_ftemp_inst_rd,
    calc_ftemp_inst_vcmax,
    calc_ftemp_kphio,
    calc_gammastar,
    calc_kmm,
    calc_ns_star,
    calc_patm,
    calc_soilmstress,
    calc_viscosity_h2o,
)
from pyrealm.pmodel.isotopes import CalcCarbonIsotopes
from pyrealm.pmodel.pmodel import (
    CalcOptimalChi,
    JmaxLimitation,
    PModel,
    PModelEnvironment,
)
from pyrealm.pmodel.subdaily import FastSlowPModel, memory_effect
