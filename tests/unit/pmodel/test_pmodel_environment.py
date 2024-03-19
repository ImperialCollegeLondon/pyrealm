"""Testing PModel Environment submodule."""

import pytest

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.pmodel.functions import (
    calc_co2_to_ca,
    calc_gammastar,
    calc_kmm,
    calc_ns_star,
)

"""Testing the intermediate PModel varaibles (kmm,gammastar, ns_star, co2_to_ca)."""


# Define test cases
@pytest.mark.parametrize(
    "tc, patm, expected_result",
    [
        (20, 101325, 46.09928),  # Test case with known inputs and expected output
        # Add more test cases here with different inputs and expected outputs
    ],
)
def test_calc_kmm(tc, patm, expected_result):
    """Test the calc_kmm function."""
    # Create an instance of PModelConst with the required attributes
    pmodel_const = PModelConst(
        bernacchi_kc25=39.97,
        bernacchi_ko25=27480,
        bernacchi_dhac=79430,
        bernacchi_dhao=36380,
    )
    # Call the function with the test inputs and the PModelConst instance
    result = calc_kmm(tc=tc, patm=patm, pmodel_const=pmodel_const)
    # Check if the result matches the expected output
    assert result == pytest.approx(expected_result, abs=1e-5)


@pytest.mark.parametrize(
    "tc, patm, expected_result",
    [
        (20, 101325, 1.12536),  # Test case with known inputs and expected output
        # Add more test cases here with different inputs and expected outputs
    ],
)
def test_calc_ns_star(tc, patm, expected_result):
    """Test the calc_ns_star function."""
    # Create an instance of CoreConst with the required attributes
    core_const = CoreConst(k_To=298.15, k_Po=101325)
    # Call the function with the test inputs and the CoreConst instance
    result = calc_ns_star(tc=tc, patm=patm, core_const=core_const)
    # Check if the result matches the expected output
    assert result == pytest.approx(expected_result, abs=1e-5)


@pytest.mark.parametrize(
    "tc, patm, expected_result",
    [
        (20, 101325, 3.33925),  # Test case with known inputs and expected output
        # Add more test cases here with different inputs and expected outputs
    ],
)
def test_calc_gammastar(tc, patm, expected_result):
    """Test the calc_gammastar function."""
    # Create instances of CoreConst and PModelConst with the required attributes
    core_const = CoreConst(k_To=298.15, k_Po=101325)
    pmodel_const = PModelConst(bernacchi_gs25_0=4.332, bernacchi_dha=37830)
    # Call the function with the test inputs and the CoreConst and PModelConst instances
    result = calc_gammastar(
        tc=tc, patm=patm, pmodel_const=pmodel_const, core_const=core_const
    )
    # Check if the result matches the expected output
    assert result == pytest.approx(expected_result, abs=1e-5)


@pytest.mark.parametrize(
    "co2, patm, expected_result",
    [
        (413.03, 101325, 41.850265),  # Test case with known inputs and expected output
        # Add more test cases here with different inputs and expected outputs
    ],
)
def test_calc_co2_to_ca(co2, patm, expected_result):
    """Test function calc_co2_to_ca."""
    # Call the function with the test inputs
    result = calc_co2_to_ca(co2=co2, patm=patm)
    # Check if the result matches the expected output
    assert result == pytest.approx(expected_result, abs=1e-6)
