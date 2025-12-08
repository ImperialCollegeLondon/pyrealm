"""Testing pmodel submodule."""

from contextlib import nullcontext as does_not_raise
from itertools import product

import numpy as np


def test_pmodel_method_combination_checking():
    """Run PModels with all possible method argument combinations.

    This test checks that the required variables from each of the methods are passed
    correctly into PModel and SubdailyPModel code.
    """

    from pyrealm.core.bounds import BoundsChecker
    from pyrealm.pmodel import (
        AcclimationModel,
        PModel,
        PModelEnvironment,
        SubdailyPModel,
    )
    from pyrealm.pmodel.arrhenius import ARRHENIUS_METHOD_REGISTRY
    from pyrealm.pmodel.jmax_limitation import JMAX_LIMITATION_CLASS_REGISTRY
    from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY
    from pyrealm.pmodel.quantum_yield import QUANTUM_YIELD_CLASS_REGISTRY

    # Get all the variables possibly required across all the methods
    required_vars = (
        [v.required_env_variables for v in ARRHENIUS_METHOD_REGISTRY.values()]
        + [v.required_env_variables for v in OPTIMAL_CHI_CLASS_REGISTRY.values()]
        + [v.required_env_variables for v in QUANTUM_YIELD_CLASS_REGISTRY.values()]
    )

    # Combine with standard PModelEnvironment variables
    all_vars = {"tc", "patm", "vpd", "co2", "fapar", "ppfd"}.union(
        {element for tpl in required_vars for element in tpl}
    )

    # Build data using bounds to give sensible constant values
    bounds = BoundsChecker()
    data = {}
    for var in all_vars:
        var_bounds = bounds._data[var]
        data[var] = np.full(
            62 * 24, fill_value=(var_bounds.upper + var_bounds.lower) / 2
        )

    env = PModelEnvironment(**data)

    # Acclimation model
    acclim = AcclimationModel(
        datetimes=np.arange(
            np.datetime64("2000-07-01"),
            np.datetime64("2000-09-01"),
            np.timedelta64(1, "h"),
        )
    )
    acclim.set_window(np.timedelta64(12, "h"), np.timedelta64(10, "m"))

    # Iterate over model combinations
    method_combinations = product(
        ARRHENIUS_METHOD_REGISTRY,
        OPTIMAL_CHI_CLASS_REGISTRY,
        QUANTUM_YIELD_CLASS_REGISTRY,
        JMAX_LIMITATION_CLASS_REGISTRY,
    )

    for arh_meth, chi_meth, phi0_meth, jmax_meth in method_combinations:
        with does_not_raise():
            _ = PModel(
                env=env,
                method_arrhenius=arh_meth,
                method_optchi=chi_meth,
                method_kphio=phi0_meth,
                method_jmaxlim=jmax_meth,
            )

            _ = SubdailyPModel(
                env=env,
                acclim_model=acclim,
                method_arrhenius=arh_meth,
                method_optchi=chi_meth,
                method_kphio=phi0_meth,
                method_jmaxlim=jmax_meth,
            )
