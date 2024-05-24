"""Test the solar functions."""

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="nu, k_e, expected",
    argvalues=[
        (np.array([0, 180, 360]), 0.0167, np.array([1.0342557, 0.9674184, 1.0342557])),
    ],
)
def test_calc_distance_factor(nu, k_e, expected):
    """Tests calc_distance_factor.

    The test values represent the range of acceptable input values of nu
    (0-360 degrees), and a typical value of k_e for the earth. This tests
    aims to confirm the correct implementation of the maths.
    """
    from pyrealm.core.solar import calc_distance_factor

    result = calc_distance_factor(nu, k_e)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="lambda_, k_eps, k_pir, expected",
    argvalues=[
        (
            np.array([-90, 0, 90]),
            23.45,
            57.29577951,
            np.array([-0.007143278, 0, 0.007143278]),
        )
    ],
)
def test_calc_declination_angle_delta(lambda_, k_eps, k_pir, expected):
    """Tests calc_declination_angle_delta.

    This test tests the maths over the applicable range of longitudes with
    representative k_eps and k_pir constants.
    """

    from pyrealm.core.solar import calc_declination_angle_delta

    result = calc_declination_angle_delta(lambda_, k_eps, k_pir)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="delta,lat, expected",
    argvalues=[
        (
            np.array([0.002, 0.002]),
            np.array([0, 75]),
            (np.array([0, 3.37172e-05]), np.array([0.999999999, 0.258819045])),
        ),
    ],
)
def test_calc_lat_delta_intermediates(delta, lat, expected):
    """Tests calc_lat_delta_intermediates.

    This test tests the maths over an applciable range of delta and latitude values.
    """

    from pyrealm.core.solar import calc_lat_delta_intermediates

    result = calc_lat_delta_intermediates(delta, lat)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="ru, rv, k_pir, expected",
    argvalues=[
        (
            np.array([0.1, 0.8]),
            np.array([0.8, 0.1]),
            np.array([0.029602951, 0.054831136]),
        )
    ],
)
def test_calc_sunset_hour_angle(ru, rv, k_pir, expected):
    """Tests calc_sunset_hour_angle.

    This test is intended to verify the maths over a range of applicable input
    values in the correct formats
    """

    from pyrealm.core.solar import calc_sunset_hour_angle

    result = calc_sunset_hour_angle(ru, rv, k_pir)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="day,n_day,expected",
    argvalues=[
        (240, 365, (231.44076437634416, 154.44076437634416)),
        (120, 365, (116.31757407861224, 39.31757407861224)),
        (240, 364, (231.86554398548128, 154.86554398548128)),
        (120, 364, (116.42440153600026, 39.42440153600026)),
    ],
)
def test_calc_heliocentric_longitudes(day, n_day, expected):
    """Tests calc_heliocentric_longitudes.

    The test values here are a small selection of values taken from feeding the inputs
    into the original function implementation. This test is primarily about ensuring the
    refactoring into pyrealm.core works.
    """
    from pyrealm.core.solar import calc_heliocentric_longitudes

    result = calc_heliocentric_longitudes(day, n_day)
    assert np.allclose(result, expected)
