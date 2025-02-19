"""Testing PModel Environment submodule."""

import json
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from _pytest.recwarn import WarningsChecker

""""Test that the pmodel environment calculates attributes match the expected values."""


@pytest.mark.parametrize(
    "vars,expected,outcome,message",
    [
        pytest.param(
            dict(
                tc=np.array([20]),
                vpd=np.array([820]),
                co2=np.array([400]),
                patm=np.array([101325.0]),
            ),
            dict(tk=293.15, ca=40.53, gammastar=3.33925, kmm=46.09928, ns_star=1.12536),
            does_not_raise(),
            None,
            id="simple",
        ),
        pytest.param(
            dict(
                tc=np.array([20]),
                vpd=np.array([820]),
                co2=np.array([400]),
                patm=np.array([101325.0]),
                theta=np.array([0.5]),
            ),
            dict(
                tk=293.15,
                ca=40.53,
                gammastar=3.33925,
                kmm=46.09928,
                ns_star=1.12536,
                theta=0.5,
            ),
            does_not_raise(),
            None,
            id="extra_theta",
        ),
        pytest.param(
            dict(
                tc=np.array([20]),
                vpd=np.array([820]),
                co2=np.array([400]),
                patm=np.array([101325.0]),
                theta=np.array([-0.5]),
            ),
            dict(
                tk=293.15,
                ca=40.53,
                gammastar=3.33925,
                kmm=46.09928,
                ns_star=1.12536,
                theta=-0.5,
            ),
            pytest.warns(UserWarning),
            "Variable 'theta' (m3 m-3) contains values outside",
            id="extra_theta_outside_bounds",
        ),
        pytest.param(
            dict(
                tc=np.array([20]),
                vpd=np.array([820]),
                co2=np.array([400]),
                patm=np.array([101325.0]),
                foobar=np.array([-0.5]),
            ),
            dict(
                tk=293.15,
                ca=40.53,
                gammastar=3.33925,
                kmm=46.09928,
                ns_star=1.12536,
                foobar=-0.5,
            ),
            pytest.warns(UserWarning),
            "Variable 'foobar' is not configured in the bounds checker",
            id="unknown variable",
        ),
        pytest.param(
            dict(
                tc=np.array([20, 20, 20]),
                vpd=np.array([820, 820]),
                co2=np.array([400]),
                patm=np.array([101325.0]),
            ),
            None,
            pytest.raises(ValueError),
            "Inputs contain arrays of different shapes.",
            id="shapes_not_congruent",
        ),
    ],
)
def test_pmodel_environment(vars, expected, outcome, message):
    """Test the PModelEnvironment class."""
    from pyrealm.pmodel.pmodel_environment import PModelEnvironment

    with outcome as context:
        env = PModelEnvironment(**vars)

        for var in expected:
            assert getattr(env, var) == pytest.approx(expected[var], abs=1e-5)

    # Check context messages
    if isinstance(context, WarningsChecker):
        assert str(context.list[0].message).startswith(message)
    elif isinstance(context, pytest.ExceptionInfo):
        assert str(context.value).startswith(message)


# Testing the boundaries of variables kmm,gammastar, ns_star, co2_to_ca.


@pytest.fixture
def function_test_data():
    """Function to load input data for testing."""
    data_path = "pyrealm_build_data/rpmodel/test_inputs.json"
    with open(data_path) as f:
        data = json.load(f)
    # Test values of forcing variables as input to functions
    return data["tc_ar"], data["patm_ar"], data["co2_ar"]


""""Test that kmm output is within bounds"""


def test_out_of_bound_output(function_test_data):
    """Function to calculate kmm."""
    from pyrealm.constants import CoreConst
    from pyrealm.pmodel.functions import calc_kmm

    tc_ar_values, patm_ar_values, co2_ar_values = function_test_data

    core_const = CoreConst()
    kmm_lower_bound = 0
    kmm_upper_bound = 1000

    for tc, patm, co2 in zip(tc_ar_values, patm_ar_values, co2_ar_values):
        result = calc_kmm(tk=tc + core_const.k_CtoK, patm=patm)

        assert np.all(result >= kmm_lower_bound), (
            f"Result for (tc={tc}, patm={patm}, co2={co2}) is out of lower bound"
        )
        assert np.all(result <= kmm_upper_bound), (
            f"Result for (tc={tc}, patm={patm}, co2={co2}) is out of upper bound"
        )


"""Test that calc_ns_star output is within bounds"""


def test_out_of_bound_output_ns_star(function_test_data):
    """Function to calculate ns_star."""
    from pyrealm.constants import CoreConst
    from pyrealm.pmodel.functions import calc_ns_star

    tc_ar_values, patm_ar_values, _ = function_test_data  # Ignore the third variable

    core_const = CoreConst(k_To=298.15, k_Po=101325)
    ns_star_lower_bound = 0
    ns_star_upper_bound = 10

    for tc, patm in zip(tc_ar_values, patm_ar_values):
        result = calc_ns_star(tc=tc, patm=patm, core_const=core_const)

        assert np.all(result >= ns_star_lower_bound), (
            f"Result for (tc={tc}, patm={patm}) is out of lower bound"
        )
        assert np.all(result <= ns_star_upper_bound), (
            f"Result for (tc={tc}, patm={patm}) is out of upper bound"
        )


"""Test that calc_gammastar output is within bounds."""


def test_out_of_bound_output_gammastar(function_test_data):
    """Function to calculate calc_gammastar."""
    from pyrealm.constants import CoreConst, PModelConst
    from pyrealm.pmodel.functions import calc_gammastar

    tc_ar_values, patm_ar_values, _ = function_test_data

    core_const = CoreConst()
    pmodel_const = PModelConst()
    gammastar_lower_bound = 0
    gammastar_upper_bound = 30

    for tc, patm in zip(tc_ar_values, patm_ar_values):
        result = calc_gammastar(
            tk=tc + core_const.k_CtoK,
            patm=patm,
            tk_ref=pmodel_const.tc_ref + core_const.k_CtoK,
        )

        assert np.all(result >= gammastar_lower_bound), (
            f"Result for (tc={tc}, patm={patm}) is out of lower bound"
        )
        assert np.all(result <= gammastar_upper_bound), (
            f"Result for (tc={tc}, patm={patm}) is out of upper bound"
        )


"""Test that calc_co2_to_ca output is within bounds"""


def test_out_of_bound_output_co2_to_ca(function_test_data):
    """Function to calculate co2_to_ca."""
    from pyrealm.pmodel.functions import calc_co2_to_ca

    _, patm_ar_values, co2_ar_values = function_test_data

    co2_to_ca_lower_bound = 0
    co2_to_ca_upper_bound = 100

    for co2, patm in zip(co2_ar_values, patm_ar_values):
        result = calc_co2_to_ca(co2=co2, patm=patm)

        assert np.all(result >= co2_to_ca_lower_bound), (
            f"Result for (co2={co2}, patm={patm}) is out of lower bound"
        )
        assert np.all(result <= co2_to_ca_upper_bound), (
            f"Result for (co2={co2}, patm={patm}) is out of upper bound"
        )
