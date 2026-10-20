"""Testing pmodel submodule."""

from contextlib import nullcontext as does_not_raise
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrealm.pmodel.arrhenius import ARRHENIUS_METHOD_REGISTRY
from pyrealm.pmodel.jmax_limitation import JMAX_LIMITATION_CLASS_REGISTRY
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY
from pyrealm.pmodel.quantum_yield import QUANTUM_YIELD_CLASS_REGISTRY


@pytest.mark.parametrize(
    argnames="method_combination",
    argvalues=(
        pytest.param(combo, id="_".join(combo))
        for combo in product(
            ARRHENIUS_METHOD_REGISTRY,
            OPTIMAL_CHI_CLASS_REGISTRY,
            QUANTUM_YIELD_CLASS_REGISTRY,
            JMAX_LIMITATION_CLASS_REGISTRY,
        )
    ),
)
def test_pmodel_method_combination_checking(method_combination):
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

    # Get all the variables possibly required across all the methods
    # NOTE: JMaxLim does not use the PModelEnv as an input - currently has no methods
    #       that require additional env variables.
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

    # Acclimation model
    acclim = AcclimationModel(
        datetimes=np.arange(
            np.datetime64("2000-07-01"),
            np.datetime64("2000-09-01"),
            np.timedelta64(1, "h"),
        )
    )
    acclim.set_window(np.timedelta64(12, "h"), np.timedelta64(10, "m"))

    arh_meth, chi_meth, phi0_meth, jmax_meth = method_combination

    with does_not_raise():
        # Get the subset of variables required for this combination
        this_combo_required_vars = (
            "tc",
            "patm",
            "vpd",
            "co2",
            "fapar",
            "ppfd",
            *ARRHENIUS_METHOD_REGISTRY[arh_meth].required_env_variables,
            *OPTIMAL_CHI_CLASS_REGISTRY[chi_meth].required_env_variables,
            *QUANTUM_YIELD_CLASS_REGISTRY[phi0_meth].required_env_variables,
        )

        # Build the combo specific environment
        env = PModelEnvironment(
            **{v: d for v, d in data.items() if v in this_combo_required_vars}
        )

        # Check the models run
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


@pytest.mark.parametrize(argnames="pmodel_type", argvalues=("standard", "subdaily"))
def test_PModel_gpp_penalty_factor(capsys, be_vie_data_components, pmodel_type):
    """Test the apply_gpp_penalty_factor mechanisms work as expected."""
    from pyrealm.pmodel import AcclimationModel, PModel, SubdailyPModel

    env, datetime, _ = be_vie_data_components.get()

    if pmodel_type == "standard":
        # Test the methods on a standard PModel
        pmodel = PModel(env)
    else:
        acclim = AcclimationModel(datetimes=datetime, allow_holdover=True)
        acclim.set_window(
            window_center=np.timedelta64(12, "h"), half_width=np.timedelta64(1, "h")
        )
        pmodel = SubdailyPModel(env=env, acclim_model=acclim)

    # Record raw GPP values
    raw_gpp = pmodel.gpp

    # Apply a penalty factor
    pmodel.apply_gpp_penalty_factor(penalty_factor=np.array([0.5]))

    # Check the gpp attribute is halved as expected
    assert_allclose(raw_gpp / 2, pmodel.gpp)

    # Check the __repr__ and summarize() methods are extended
    assert "penalized" in repr(pmodel)
    pmodel.summarize()
    out = capsys.readouterr().out
    assert "gpp_penalty_factor" in out

    # Revert to unpenalised prediction
    pmodel.remove_gpp_penalty_factor()

    # Values are back to original
    assert_allclose(raw_gpp, pmodel.gpp)

    # Check the __repr__ and summarize() methods are extended
    assert "penalized" not in repr(pmodel)
    pmodel.summarize()
    out = capsys.readouterr().out
    assert "gpp_penalty_factor" not in out
