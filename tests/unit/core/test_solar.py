"""Test the solar functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# @pytest.fixture(scope="module")
# def CoreConst_fixture():
# """Sets up core constants dataclass."""
# const = CoreConst()
# return CoreConst()


@pytest.mark.parametrize(
    argnames="nu, expected",
    argvalues=[
        (np.array([166.097934]), np.array([0.968381])),
    ],
)
def test_calculate_distance_factor(nu, expected):
    """Tests test_calculate_distance_factor.

    This tests aims to confirm the correct implementation of the maths.
    """
    from pyrealm.core.solar import calculate_distance_factor

    result = calculate_distance_factor(nu)

    assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    argnames="lambda_, expected",
    argvalues=[
        (
            np.array([89.097934]),
            np.array([23.436921]),
        )
    ],
)
def test_calculate_solar_declination_angle(lambda_, expected):
    """Tests calculate_solar_declination_angle.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calculate_solar_declination_angle

    result = calculate_solar_declination_angle(lambda_)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="delta,latitude, expected",
    argvalues=[
        (
            np.array([23.436921]),
            np.array([37.7]),
            (np.array([0.243228277]), np.array([0.725946417])),
        ),
    ],
)
def test_calculate_ru_rv_intermediates(delta, latitude, expected):
    """Tests calculate_ru_rv_intermediates.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calculate_ru_rv_intermediates

    result = calculate_ru_rv_intermediates(delta, latitude)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="delta, latitude, expected",
    argvalues=[
        (
            np.array([23.436921]),
            np.array([37.7]),
            np.array([109.575573]),
        )
    ],
)
def test_calc_sunset_hour_angle(delta, latitude, expected):
    """Tests calc_sunset_hour_angle.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import (
        _calculate_sunset_hour_angle,
        calculate_ru_rv_intermediates,
        calculate_sunset_hour_angle,
    )

    result = calculate_sunset_hour_angle(declination=delta, latitude=latitude)

    assert_allclose(result, expected)

    ru, rv = calculate_ru_rv_intermediates(declination=delta, latitude=latitude)

    result = _calculate_sunset_hour_angle(ru=ru, rv=rv)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="dr, hs, delta, latitude, expected",
    argvalues=[
        (
            np.array([0.968381]),
            np.array([109.575573]),
            np.array([23.436921]),
            np.array([37.7]),
            np.array([41646763]),
        )
    ],
)
def test_calculate_daily_solar_radiation(dr, hs, delta, latitude, expected):
    """Tests calculate_daily_solar_radiation.

    This test is intended to verify the implemented maths
    """

    from pyrealm.core.solar import (
        _calculate_daily_solar_radiation,
        calculate_daily_solar_radiation,
        calculate_ru_rv_intermediates,
    )

    result = calculate_daily_solar_radiation(
        distance_ratio=dr, sunset_hour_angle=hs, declination=delta, latitude=latitude
    )

    assert_allclose(result, expected, rtol=1e-6)

    ru, rv = calculate_ru_rv_intermediates(declination=delta, latitude=latitude)

    result = _calculate_daily_solar_radiation(
        ru=ru, rv=rv, distance_ratio=dr, sunset_hour_angle=hs
    )

    assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    argnames="sf, elv, expected",
    argvalues=[(np.array([1.0]), np.array([142]), np.array([0.752844]))],
)
def test_calculate_transmissivity(sf, elv, expected):
    """Tests calculate_transmissivity.

    This test is intended to verify the implemented maths
    """

    from pyrealm.core.solar import calculate_transmissivity

    result = calculate_transmissivity(sf, elv)

    assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    argnames="tau, ra_d, expected",
    argvalues=[(np.array([0.752844]), np.array([41646763]), np.array([62.042300]))],
)
def test_calculate_ppfd_from_tau_rd(tau, ra_d, expected):
    """Tests calc_ppfd_from_tau_ra_d.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calculate_ppfd_from_tau_rd

    result = calculate_ppfd_from_tau_rd(transmissivity=tau, daily_solar_radiation=ra_d)

    assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    argnames="sf, elv, latitude, julian_day, n_days, expected",
    argvalues=[
        (
            np.array([1.0]),
            np.array([142]),
            np.array([37.7]),
            np.array([172]),
            np.array([366]),
            np.array([62.042300]),
        )
    ],
)
def test_calculate_ppfd(
    sf,
    elv,
    latitude,
    julian_day,
    n_days,
    expected,
):
    """Tests calculate_ppfd.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calculate_ppfd

    result = calculate_ppfd(sf, elv, latitude, julian_day, n_days)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="sf, tc, expected",
    argvalues=[(np.array([1.0]), np.array([23.0]), np.array([84.000000]))],
)
def test_calculate_net_longwave_radiation(sf, tc, expected):
    """Tests calc_net_longwave_radiation.

    This test is intended to verify the implemented maths.
    """
    from pyrealm.core.solar import calculate_net_longwave_radiation

    result = calculate_net_longwave_radiation(sf, tc)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="tau, dr, expected",
    argvalues=[
        (
            np.array([0.752844]),
            np.array([0.968381]),
            np.array([823.4242375]),
        )
    ],
)
def test_calculate_rw_intermediate(tau, dr, expected):
    """Test calculate_rw_intermediate.

    This test is intended to verify the implemented maths.
    """
    from pyrealm.core.solar import calculate_rw_intermediate

    result = calculate_rw_intermediate(transmissivity=tau, distance_ratio=dr)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="inputs, expected",
    argvalues=[
        (
            {
                "net_longwave_radiation": np.array([84.000000]),
                "transmissivity": np.array([0.752844]),
                "distance_ratio": np.array([0.968381]),
                "declination": np.array([23.436921]),
                "latitude": np.array([37.7]),
            },
            np.array([101.217016]),
        )
    ],
)
def test_calculate_net_radiation_crossover_hour_angle(inputs, expected):
    """Tests calculate_net_radiation_crossover_hour_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import (
        _calculate_net_radiation_crossover_hour_angle,
        calculate_net_radiation_crossover_hour_angle,
        calculate_ru_rv_intermediates,
        calculate_rw_intermediate,
    )

    result = calculate_net_radiation_crossover_hour_angle(**inputs)

    assert_allclose(result, expected)

    ru, rv = calculate_ru_rv_intermediates(
        declination=inputs["declination"], latitude=inputs["latitude"]
    )
    rw = calculate_rw_intermediate(
        transmissivity=inputs["transmissivity"], distance_ratio=inputs["distance_ratio"]
    )

    result = _calculate_net_radiation_crossover_hour_angle(
        ru=ru,
        rv=rv,
        rw=rw,
        net_longwave_radiation=inputs["net_longwave_radiation"],
    )

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="inputs, expected",
    argvalues=[
        (
            {
                "net_longwave_radiation": np.array([84.000000]),
                "crossover_hour_angle": np.array([101.217016]),
                "declination": np.array([23.436921]),
                "latitude": np.array([37.7]),
                "transmissivity": np.array([0.752844]),
                "distance_ratio": np.array([0.968381]),
            },
            np.array([21774962.26810856]),
        )
    ],
)
def test_daytime_net_radiation(inputs, expected):
    """Tests calculation of net daytime radiation.

    This test orginally had an expected value of [21774953], which required rtol=1e-6 in
    the assertions. It isn't clear where that test value came from, but the current one
    allows for more exact tests, which is probably better for trapping small errors.
    """

    from pyrealm.core.solar import (
        _calculate_daytime_net_radiation,
        calculate_daytime_net_radiation,
        calculate_ru_rv_intermediates,
        calculate_rw_intermediate,
    )

    result = calculate_daytime_net_radiation(**inputs)

    assert_allclose(result, expected)

    ru, rv = calculate_ru_rv_intermediates(
        declination=inputs["declination"], latitude=inputs["latitude"]
    )
    rw = calculate_rw_intermediate(
        transmissivity=inputs["transmissivity"], distance_ratio=inputs["distance_ratio"]
    )

    result = _calculate_daytime_net_radiation(
        ru=ru,
        rv=rv,
        rw=rw,
        crossover_hour_angle=inputs["crossover_hour_angle"],
        net_longwave_radiation=inputs["net_longwave_radiation"],
    )

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="inputs, expected",
    argvalues=[
        (
            {
                "net_longwave_radiation": np.array([84.000000]),
                "crossover_hour_angle": np.array([101.217016]),
                "sunset_hour_angle": np.array([109.575573]),
                "declination": np.array([23.436921]),
                "latitude": np.array([37.7]),
                "transmissivity": np.array([0.752844]),
                "distance_ratio": np.array([0.968381]),
            },
            np.array([-3009150]),
        ),
    ],
)
def test_nightime_net_radiation(inputs, expected):
    """Tests calculation of net nighttime radiation."""

    from pyrealm.core.solar import (
        _calculate_nighttime_net_radiation,
        calculate_nighttime_net_radiation,
        calculate_ru_rv_intermediates,
        calculate_rw_intermediate,
    )

    result = calculate_nighttime_net_radiation(**inputs)

    assert_allclose(result, expected)

    ru, rv = calculate_ru_rv_intermediates(
        declination=inputs["declination"], latitude=inputs["latitude"]
    )
    rw = calculate_rw_intermediate(
        transmissivity=inputs["transmissivity"], distance_ratio=inputs["distance_ratio"]
    )

    result = _calculate_nighttime_net_radiation(
        ru=ru,
        rv=rv,
        rw=rw,
        sunset_hour_angle=inputs["sunset_hour_angle"],
        crossover_hour_angle=inputs["crossover_hour_angle"],
        net_longwave_radiation=inputs["net_longwave_radiation"],
    )

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="day,n_day,expected",
    argvalues=[
        (240, 365, (231.44076437634416, 154.44076437634416)),
        (120, 365, (116.31757407861224, 39.31757407861224)),
        (240, 364, (231.86554398548128, 154.86554398548128)),
        (120, 364, (116.42440153600026, 39.42440153600026)),
    ],
)
def test_calculate_heliocentric_longitudes(day, n_day, expected):
    """Tests calculate_heliocentric_longitudes.

    The test values here are a small selection of values taken from feeding the inputs
    into the original function implementation. This test is primarily about ensuring the
    refactoring into pyrealm.core works.
    """
    from pyrealm.core.solar import calculate_heliocentric_longitudes

    result = calculate_heliocentric_longitudes(day, n_day)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="latitude, longitude, year_date_time, expected",
    argvalues=[
        (
            -35.058333,
            147.34167,
            np.array([np.datetime64("1995-10-25T10:30")]),
            np.array([1.0615713]),
        )
    ],
)
def test_calculate_solar_elevation(latitude, longitude, year_date_time, expected):
    """Tests calculate_solar_elevation.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.calendar import LocationDateTime
    from pyrealm.core.solar import calculate_solar_elevation

    site_obs_data = LocationDateTime(
        latitude=latitude,
        longitude=longitude,
        year_date_time=year_date_time,
    )

    result = calculate_solar_elevation(site_obs_data)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="td, expected",
    argvalues=[(np.array([298]), -0.22708144)],
)
def test_calculate_solar_declination(td, expected):
    """Tests calculate_solar_declination.

    This test is intended to verify the implemented maths.
    """

    # This test is intended to verify the implemented maths.

    from pyrealm.core.solar import calculate_solar_declination

    # Const = CoreConst()

    result = calculate_solar_declination(td)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="julian_day, expected",
    argvalues=[(np.array([298]), 5.11261928)],
)
def test_calculate_day_angle(julian_day, expected):
    """Tests calculate_day_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calculate_day_angle

    result = calculate_day_angle(julian_day)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="day_angle, expected",
    argvalues=[(np.array([5.11]), 15.99711625)],
)
def test_calculate_equation_of_time(day_angle, expected):
    """Tests calculate_equation_of_time.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calculate_equation_of_time

    result = calculate_equation_of_time(day_angle)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="longitude, standard_meridian, E_t, expected",
    argvalues=[(147.34167, 150, 16.01, np.array([11.910388666666668]))],
)
def test_calculate_solar_noon(longitude, standard_meridian, E_t, expected):
    """Tests calculate_solar_noon.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calculate_solar_noon

    result = calculate_solar_noon(longitude, E_t, standard_meridian)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="t, t0, expected",
    argvalues=[(np.array([10.5]), np.array([11.91]), np.array([-0.36913714]))],
)
def test_calculate_local_hour_angle(t, t0, expected):
    """Tests calculate_local_hour_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calculate_local_hour_angle

    result = calculate_local_hour_angle(t, t0)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="latitude, declination, hour_angle, expected",
    argvalues=[
        (np.array([-0.61]), np.array([-0.23]), np.array([-0.37]), np.array([1.0647289]))
    ],
)
def test_calculate_solar_elevation_angle(latitude, declination, hour_angle, expected):
    """Tests calculate_solar_elevation_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calculate_solar_elevation_angle

    result = calculate_solar_elevation_angle(latitude, declination, hour_angle)

    assert_allclose(result, expected)
