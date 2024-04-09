"""Testing PModel Environment submodule."""


import json

import numpy as np
import pytest

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.pmodel.functions import (
    calc_co2_to_ca,
    calc_gammastar,
    calc_kmm,
    calc_ns_star,
)
from pyrealm.pmodel.pmodel_environment import PModelEnvironment

""""Test that the pmodel environment calculates attributes match the expected values."""


@pytest.mark.parametrize(
    "tc,vpd,co2, patm, expected_ca, expected_gammastar, expected_kmm, expected_ns_star",
    [
        (
            np.array([20]),
            np.array([820]),
            np.array([400]),
            np.array([101325.0]),
            40.53,
            3.33925,
            46.09928,
            1.12536,
        ),
    ],
)
def test_pmodel_environment(
    tc, vpd, co2, patm, expected_ca, expected_gammastar, expected_kmm, expected_ns_star
):
    """Test the PModelEnvironment class."""
    env = PModelEnvironment(tc=tc, vpd=vpd, co2=co2, patm=patm)

    assert env.ca == pytest.approx(expected_ca, abs=1e-5)
    assert env.gammastar == pytest.approx(expected_gammastar, abs=1e-5)
    assert env.kmm == pytest.approx(expected_kmm, abs=1e-5)
    assert env.ns_star == pytest.approx(expected_ns_star, abs=1e-5)


"""Testing the boundries of variables kmm,gammastar, ns_star, co2_to_ca."""


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
    tc_ar_values, patm_ar_values, co2_ar_values = function_test_data

    pmodel_const = PModelConst()
    kmm_lower_bound = 0
    kmm_upper_bound = 1000

    for tc, patm, co2 in zip(tc_ar_values, patm_ar_values, co2_ar_values):
        result = calc_kmm(tc=tc, patm=patm, pmodel_const=pmodel_const)

        assert np.all(
            result >= kmm_lower_bound
        ), f"Result for (tc={tc}, patm={patm}, co2={co2}) is out of lower bound"
        assert np.all(
            result <= kmm_upper_bound
        ), f"Result for (tc={tc}, patm={patm}, co2={co2}) is out of upper bound"


"""Test that calc_ns_star output is within bounds"""


def test_out_of_bound_output_ns_star(function_test_data):
    """Function to calculate ns_star."""
    tc_ar_values, patm_ar_values, _ = function_test_data  # Ignore the third variable

    core_const = CoreConst(k_To=298.15, k_Po=101325)
    ns_star_lower_bound = 0
    ns_star_upper_bound = 10

    for tc, patm in zip(tc_ar_values, patm_ar_values):
        result = calc_ns_star(tc=tc, patm=patm, core_const=core_const)

        assert np.all(
            result >= ns_star_lower_bound
        ), f"Result for (tc={tc}, patm={patm}) is out of lower bound"
        assert np.all(
            result <= ns_star_upper_bound
        ), f"Result for (tc={tc}, patm={patm}) is out of upper bound"


"""Test that calc_gammastar output is within bounds."""


def test_out_of_bound_output_gammastar(function_test_data):
    """Function to calculate calc_gammastar."""
    tc_ar_values, patm_ar_values, _ = function_test_data

    core_const = CoreConst(k_To=298.15, k_Po=101325)
    pmodel_const = PModelConst(bernacchi_gs25_0=4.332, bernacchi_dha=37830)
    gammastar_lower_bound = 0
    gammastar_upper_bound = 30

    for tc, patm in zip(tc_ar_values, patm_ar_values):
        result = calc_gammastar(
            tc=tc, patm=patm, pmodel_const=pmodel_const, core_const=core_const
        )

        assert np.all(
            result >= gammastar_lower_bound
        ), f"Result for (tc={tc}, patm={patm}) is out of lower bound"
        assert np.all(
            result <= gammastar_upper_bound
        ), f"Result for (tc={tc}, patm={patm}) is out of upper bound"


"""Test that calc_co2_to_ca output is within bounds"""


def test_out_of_bound_output_co2_to_ca(function_test_data):
    """Function to calculate co2_to_ca."""
    _, patm_ar_values, co2_ar_values = function_test_data

    co2_to_ca_lower_bound = 0
    co2_to_ca_upper_bound = 100

    for co2, patm in zip(co2_ar_values, patm_ar_values):
        result = calc_co2_to_ca(co2=co2, patm=patm)

        assert np.all(
            result >= co2_to_ca_lower_bound
        ), f"Result for (co2={co2}, patm={patm}) is out of lower bound"
        assert np.all(
            result <= co2_to_ca_upper_bound
        ), f"Result for (co2={co2}, patm={patm}) is out of upper bound"
