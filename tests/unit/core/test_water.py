"""Test the pmodel functions.

TODO - note that there are parallel tests in test_pmodel that benchmark against the
rpmodel outputs and test a wider range of inputs. Those could be moved here. These tests
check the size of outputs and that the results meet a simple benchmark value.
"""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray


def test_density_registration():
    """Test density registration checks signatures and extends registry."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.water import DENSITY_METHODS, register_density_method

    with pytest.raises(RuntimeError):

        @register_density_method("bad_param_name")
        def bad_param_name(
            tk: NDArray[np.floating], patm: NDArray[np.floating], core_const: CoreConst
        ) -> NDArray[np.floating]:
            return np.ones(1)

        assert "bad_param_name" not in DENSITY_METHODS

    with pytest.raises(RuntimeError):

        @register_density_method("bad_annotation")
        def bad_annotation(
            tc: NDArray[np.integer], patm: NDArray[np.integer], core_const: CoreConst
        ) -> NDArray[np.floating]:
            return np.ones(1)

        assert "bad_annotation" not in DENSITY_METHODS

    with does_not_raise():

        @register_density_method("good")
        def good(
            tc: NDArray[np.floating], patm: NDArray[np.floating], core_const: CoreConst
        ) -> NDArray[np.floating]:
            return np.ones(1)

        assert "good" in DENSITY_METHODS


def test_viscosity_registration():
    """Test viscosity registration checks signatures and extends registry."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.water import VISCOSITY_METHODS, register_viscosity_method

    with pytest.raises(RuntimeError):

        @register_viscosity_method("bad_param_name")
        def bad_param_name(
            tc: NDArray[np.floating], patm: NDArray[np.floating], core_const: CoreConst
        ) -> NDArray[np.floating]:
            return np.ones(1)

        assert "bad_param_name" not in VISCOSITY_METHODS

    with pytest.raises(RuntimeError):

        @register_viscosity_method("bad_annotation")
        def bad_annotation(
            tk: NDArray[np.integer], patm: NDArray[np.integer], core_const: CoreConst
        ) -> NDArray[np.floating]:
            return np.ones(1)

        assert "bad_annotation" not in VISCOSITY_METHODS

    with does_not_raise():

        @register_viscosity_method("good")
        def good(
            tk: NDArray[np.floating], patm: NDArray[np.floating], core_const: CoreConst
        ) -> NDArray[np.floating]:
            return np.ones(1)

        assert "good" in VISCOSITY_METHODS


@pytest.mark.parametrize(
    argnames="method, expected",
    argvalues=(
        pytest.param("fisher", 998.206, id="fisher"),
        pytest.param("chen", 998.25, id="chen"),
        pytest.param("jones_harris_eq8", 998.201, id="jones_harris_eq8"),
        pytest.param("jones_harris_eq6", 998.201, id="jones_harris_eq6"),
        pytest.param("kell", 997.936, id="kell"),
    ),
)
@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        pytest.param((1,), id="1D"),
        pytest.param((6, 9), id="2D"),
        pytest.param((4, 7, 3), id="3D"),
    ],
)
def test_calc_density_h20_methods(method, expected, shape):
    """Test the density methods.

    The test runs both directly using the function from the registry and via the core
    constants setting.
    """

    from pyrealm.constants import CoreConst
    from pyrealm.core.water import DENSITY_METHODS, calculate_density_h2o

    # Get the function directly from the registry and run it
    func = DENSITY_METHODS[method]

    rho = func(tc=np.full(shape, fill_value=20), patm=np.full(shape, fill_value=101325))

    assert_allclose(rho.round(3), np.full(shape, fill_value=expected))

    # Configure the method through CoreConst and run the wrapper function.
    rho = calculate_density_h2o(
        tc=np.full(shape, fill_value=20),
        patm=np.full(shape, fill_value=101325),
        core_const=CoreConst(water_density_method=method),
    )

    assert_allclose(rho.round(3), np.full(shape, fill_value=expected))


@pytest.mark.parametrize(
    argnames="method, expected",
    argvalues=(
        pytest.param("vogel", 0.00100353, id="vogel"),
        pytest.param("viswanath_natarajan", 0.00100576, id="viswanath_natarajan"),
        pytest.param("girifalco", 0.00100742, id="girifalco"),
        pytest.param("reid", 0.00101766, id="reid"),
        pytest.param("daubert_danner", 0.00103321, id="daubert_danner"),
        pytest.param("huber", 0.0010016, id="huber"),
    ),
)
@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        pytest.param((1,), id="1D"),
        pytest.param((6, 9), id="2D"),
        pytest.param((4, 7, 3), id="3D"),
    ],
)
def test_calculate_viscosity_h20_methods(method, expected, shape):
    """Test the viscosity methods.

    The test runs both directly using the function from the registry and via the core
    constants setting.
    """

    from pyrealm.constants import CoreConst
    from pyrealm.core.water import VISCOSITY_METHODS, calculate_viscosity_h2o

    # Get the function directly from the registry and run it
    func = VISCOSITY_METHODS[method]

    mu = func(
        tk=np.full(shape, fill_value=293.15), patm=np.full(shape, fill_value=101325)
    )

    assert_allclose(mu.round(8), np.full(shape, fill_value=expected))

    # Configure the method through CoreConst and run the wrapper function.
    mu = calculate_viscosity_h2o(
        tk=np.full(shape, fill_value=293.15),
        patm=np.full(shape, fill_value=101325),
        core_const=CoreConst(water_viscosity_method=method),
    )

    assert_allclose(mu.round(8), np.full(shape, fill_value=expected))


def test_calculate_water_molar_volume():
    """Simple sense check that molar volume at standard conditions ~= molar mass."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.water import calculate_water_molar_volume

    assert_allclose(
        calculate_water_molar_volume(tc=np.array([0]), patm=np.array([101325])),
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
        convert_water_mm_to_moles(
            water_mm=np.array([1]), tc=np.array([0]), patm=np.array([101325])
        ),
        55.508,
        rtol=1e-5,
    )

    # At 0°C and 101325 Pa, one mole of water is ~18 g (18 cm3, 0.018 mm m-2).
    # So, 1 mol = 0.018 mm
    assert_allclose(
        convert_water_moles_to_mm(
            water_moles=np.array([1]), tc=np.array([0]), patm=np.array([101325])
        ),
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
    patm = np.array([101325])

    with pytest.raises(ValueError):
        convert_water_mm_to_moles(water_mm, tc, patm)

    with pytest.raises(ValueError):
        convert_water_mm_to_moles(water_moles, tc, patm)
