"""Test the pmodel functions.

TODO - note that there are parallel tests in test_pmodel that benchmark against the
rpmodel outputs and test a wider range of inputs. Those could be moved here. These tests
check the size of outputs and that the results meet a simple benchmark value.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_density_h20_fisher(shape):
    """Test the fisher method."""
    from pyrealm.core.water import calc_density_h2o_fisher

    rho = calc_density_h2o_fisher(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(rho.round(3), np.full(shape, fill_value=998.206))


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_density_h20_chen(shape):
    """Test the chen method."""
    from pyrealm.core.water import calc_density_h2o_chen

    rho = calc_density_h2o_chen(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(rho.round(3), np.full(shape, fill_value=998.25))


@pytest.mark.parametrize(
    argnames="const_args, exp",
    argvalues=[
        pytest.param(None, 998.206, id="defaults"),
        pytest.param("fisher", 998.206, id="fisher_via_const"),
        pytest.param("chen", 998.25, id="chen_via_const"),
    ],
)
def test_calc_density_h20(const_args, exp):
    """Test the wrapper method dispatches as expected."""
    from pyrealm.constants import CoreConst
    from pyrealm.core.water import calc_density_h2o

    args = {}

    if const_args is not None:
        args["core_const"] = CoreConst(water_density_method=const_args)

    rho = calc_density_h2o(20, 101325, **args)

    assert rho.round(3) == exp


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_viscosity_h20(shape):
    """Test the viscosity calculation."""
    from pyrealm.core.water import calc_viscosity_h2o

    eta = calc_viscosity_h2o(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(eta.round(7), np.full(shape, fill_value=0.0010016))


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_viscosity_h20_matrix(shape):
    """Test the viscosity calculation."""
    from pyrealm.core.water import calc_viscosity_h2o_matrix

    eta = calc_viscosity_h2o_matrix(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(eta.round(7), np.full(shape, fill_value=0.0010016))


def test_calculate_water_molar_volume():
    """Simple sense check that molar volume at standard conditions ~= molar mass."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.water import calculate_water_molar_volume

    assert_allclose(
        calculate_water_molar_volume(tc=0, patm=101325),
        CoreConst.k_water_molmass,
        rtol=1e-3,
    )


def test_convert_water_benchmark():
    """Test water conversion functions.

    Approximate benchmarking of convert_water_mm_to_moles and convert_water_moles_to_mm
    against real world values. Further testing below looks at round trip between the two
    functions, so this checks that the values are real world sensible.
    """

    from pyrealm.core.water import convert_water_mm_to_moles, convert_water_moles_to_mm

    # At 0°C and 101325 Pa, one mole of water is ~18 g (18 cm3, 0.018 mm m-2).
    # So, 1 mm m2 = 1 / 0.018 = ~55 moles.
    assert_allclose(
        convert_water_mm_to_moles(water_mm=1, tc=0, patm=101325),
        55.508,
        rtol=1e-5,
    )

    # At 0°C and 101325 Pa, one mole of water is ~18 g (18 cm3, 0.018 mm m-2).
    # So, 1 mol = 0.018 mm
    assert_allclose(
        convert_water_moles_to_mm(water_moles=1, tc=0, patm=101325),
        0.018015,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "water_mm, tc, patm, expected_fisher, expected_chen",
    [
        pytest.param(
            np.array([0, 1, 10, 1, 1, 1]),
            np.array([20, 20, 20, 0, 20, 20]),
            np.array([101325, 101325, 101325, 101325, 90000, 110000]),
            np.array([0.0, 55.417139, 554.171387, 55.507874, 55.41685, 55.41736]),
            np.array([0.0, 55.419629, 554.196289, 55.510708, 55.419341, 55.419849]),
            id="array_values",
        )
    ],
)
def test_convert_water_values(water_mm, tc, patm, expected_fisher, expected_chen):
    """Test the convert_water_mm_to_moles and convert_water_moles_to_mm function."""
    from pyrealm.constants import CoreConst
    from pyrealm.core.water import convert_water_mm_to_moles, convert_water_moles_to_mm

    # Fisher
    fisher_const = CoreConst(water_density_method="fisher")

    moles_water_fisher = convert_water_mm_to_moles(
        water_mm, tc, patm, core_const=fisher_const
    )

    # Test forward and back conversion
    assert_allclose(moles_water_fisher, expected_fisher, rtol=1e-5)
    assert_allclose(
        convert_water_moles_to_mm(
            moles_water_fisher, tc, patm, core_const=fisher_const
        ),
        water_mm,
        rtol=1e-4,
    )

    # Chen
    chen_const = CoreConst(water_density_method="chen")

    moles_water_chen = convert_water_mm_to_moles(
        water_mm, tc, patm, core_const=chen_const
    )

    # Test forward and back conversion
    assert_allclose(moles_water_chen, expected_chen, rtol=1e-5)
    assert_allclose(
        convert_water_moles_to_mm(moles_water_fisher, tc, patm, core_const=chen_const),
        water_mm,
        rtol=1e-4,
    )


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_convert_water(shape):
    """Test the water conversion functions with different shapes."""
    from pyrealm.core.water import convert_water_mm_to_moles, convert_water_moles_to_mm

    water_mm = np.full(shape, fill_value=1)
    tc = np.full(shape, fill_value=20)
    patm = np.full(shape, fill_value=101325)

    # Test mm to moles
    moles_water = convert_water_mm_to_moles(water_mm=water_mm, tc=tc, patm=patm)
    assert_allclose(moles_water, np.full(shape, fill_value=55.41713669719267))

    # Test reverse direction
    assert_allclose(convert_water_moles_to_mm(moles_water, tc=tc, patm=patm), water_mm)


def test_convert_water_invalid_input():
    """Test the convert_water_mm_to_moles function with invalid input."""
    from pyrealm.core.water import convert_water_mm_to_moles

    # Input shapes not equal or scalar
    water_mm = np.array([1, 2])
    water_moles = np.array([1, 2])
    tc = np.array([0, 5, 20])
    patm = 101325

    with pytest.raises(ValueError):
        convert_water_mm_to_moles(water_mm, tc, patm)

    with pytest.raises(ValueError):
        convert_water_mm_to_moles(water_moles, tc, patm)
