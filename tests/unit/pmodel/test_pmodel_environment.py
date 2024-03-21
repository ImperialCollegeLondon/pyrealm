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


"""Testing the boundries of variables (kmm,gammastar, ns_star, co2_to_ca)."""

with open("pyrealm_build_data/rpmodel/test_inputs.json") as f:
    data = json.load(f)

# Test values of forcing variables as input to functions
tc_ar_values = data["tc_ar"]
patm_ar_values = data["patm_ar"]
co2_ar_values = data["co2_ar"]


""""Test that kmm output is within bounds"""
kmm_lower_bound = 0
kmm_upper_bound = 1000


@pytest.mark.parametrize(
    "tc, patm", [(tc, patm) for tc in tc_ar_values for patm in patm_ar_values]
)
def test_out_of_bound_output(tc, patm):
    """Function to calulate kmm."""
    pmodel_const = PModelConst(
        bernacchi_kc25=39.97,
        bernacchi_ko25=27480,
        bernacchi_dhac=79430,
        bernacchi_dhao=36380,
    )

    result = calc_kmm(tc=tc, patm=patm, pmodel_const=pmodel_const)

    assert np.all(
        result >= kmm_lower_bound
    ), f"Result for (tc={tc}, patm={patm}) is out of lower bound"
    assert np.all(
        result <= kmm_upper_bound
    ), f"Result for (tc={tc}, patm={patm}) is out of upper bound"


"""Test that calc_ns_star output is within bounds"""
ns_star_lower_bound = 0
ns_star_upper_bound = 10


@pytest.mark.parametrize(
    "tc, patm", [(tc, patm) for tc in tc_ar_values for patm in patm_ar_values]
)
def test_out_of_bound_output_ns_star(tc, patm):
    """Function to calculate ns_star."""
    core_const = CoreConst(k_To=298.15, k_Po=101325)

    result = calc_ns_star(tc=tc, patm=patm, core_const=core_const)

    assert np.all(
        result >= ns_star_lower_bound
    ), f"Result for (tc={tc}, patm={patm}) is out of lower bound"
    assert np.all(
        result <= ns_star_upper_bound
    ), f"Result for (tc={tc}, patm={patm}) is out of upper bound"


"""Test that calc_gammastar output is within bounds."""
gammastar_lower_bound = 0
gammastar_upper_bound = 30


@pytest.mark.parametrize(
    "tc, patm", [(tc, patm) for tc in tc_ar_values for patm in patm_ar_values]
)
def test_out_of_bound_output_gammastar(tc, patm):
    """Function to calculate calc_gammastar."""
    core_const = CoreConst(k_To=298.15, k_Po=101325)
    pmodel_const = PModelConst(bernacchi_gs25_0=4.332, bernacchi_dha=37830)

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
co2_to_ca_lower_bound = 0
co2_to_ca_upper_bound = 100


@pytest.mark.parametrize(
    "co2, patm", [(co2, patm) for co2 in co2_ar_values for patm in patm_ar_values]
)
def test_out_of_bound_output_co2_to_ca(co2, patm):
    """Function to calculate co2_to_ca."""
    result = calc_co2_to_ca(co2=co2, patm=patm)

    assert np.all(
        result >= co2_to_ca_lower_bound
    ), f"Result for (co2={co2}, patm={patm}) is out of lower bound"
    assert np.all(
        result <= co2_to_ca_upper_bound
    ), f"Result for (co2={co2}, patm={patm}) is out of upper bound"
