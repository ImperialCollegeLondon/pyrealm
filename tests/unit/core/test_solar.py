"""Test the solar functions."""

import numpy as np
import pytest

from pyrealm.constants.core_const import CoreConst


@pytest.fixture(autouse=True, scope="module")
def CoreConst_fixture():
    """Sets up core constnats dataclass."""

    return CoreConst()


@pytest.mark.parametrize(
    argnames="nu, k_e, expected",
    argvalues=[
        (np.array([166.097934]), np.array([0.968381])),
    ],
)
def test_calc_distance_factor(nu, expected):
    """Tests calc_distance_factor.

    This tests aims to confirm the correct implementation of the maths.
    """
    from pyrealm.core.solar import calc_distance_factor

    result = calc_distance_factor(nu, CoreConst_fixture)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="lambda_, expected",
    argvalues=[
        (
            np.array([89.097934]),
            np.array([23.436921]),
        )
    ],
)
def test_calc_declination_angle_delta(lambda_, expected):
    """Tests calc_declination_angle_delta.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calc_declination_angle_delta

    result = calc_declination_angle_delta(lambda_, CoreConst_fixture)

    assert np.allclose(result, expected)


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
def test_calc_lat_delta_intermediates(delta, latitude, expected):
    """Tests calc_lat_delta_intermediates.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calc_lat_delta_intermediates

    result = calc_lat_delta_intermediates(delta, latitude)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="nu, k_e, expected",
    argvalues=[
        (np.array([166.097934]), 0.0167, np.array([0.968381])),
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
            np.array([89.097934]),
            23.45,
            57.29577951,
            np.array([23.436921]),
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
            np.array([23.436921]),
            np.array([37.7]),
            (np.array([0.243228277]), np.array([0.725946417])),
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
            np.array([23.436921]),
            np.array([37.7]),
            np.array([109.575573]),
        )
    ],
)
def test_calc_sunset_hour_angle(delta, latitude, k_pir, expected):
    """Tests calc_sunset_hour_angle.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calc_sunset_hour_angle

    result = calc_sunset_hour_angle(delta, latitude, CoreConst_fixture)

    assert np.allclose(result, expected)


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
def test_calc_daily_solar_radiation(dr, hs, delta, latitude, expected):
    """Tests calc_daily_solar_radiation.

    This test is intended to verify the implemented maths
    """

    from pyrealm.core.solar import calc_daily_solar_radiation

    result = calc_daily_solar_radiation(dr, hs, delta, latitude, CoreConst_fixture)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="sf, elv, expected",
    argvalues=[(np.array([1.0]), np.array([142]), np.array([0.752844]))],
)
def test_calc_transmissivity(sf, elv, expected):
    """Tests calc_transmissivity.

    This test is intended to verify the implemented maths
    """

    from pyrealm.core.solar import calc_transmissivity

    result = calc_transmissivity(sf, elv, CoreConst_fixture)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="k_fFEC, k_alb_vis, tau, ra_d, expected",
    argvalues=[(np.array([0.752844]), np.array([41646763]), np.array([62.042300]))],
)
def test_calc_ppfd_from_tau_ra_d(tau, ra_d, expected):
    """Tests calc_ppfd_from_tau_ra_d.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calc_ppfd_from_tau_ra_d

    result = calc_ppfd_from_tau_ra_d(tau, ra_d, CoreConst_fixture)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="sf, elv, latitude, julian_day, n_days, k_e, k_eps, k_pir, k_d, k_Gsc,\
    k_fFec, k_alb_vis, k_c, expected",
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
def test_calc_ppfd(
    sf,
    elv,
    latitude,
    julian_day,
    n_days,
    expected,
):
    """Tests calc_ppfd.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calc_ppfd

    result = calc_ppfd(sf, elv, latitude, julian_day, n_days, CoreConst_fixture)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="sf, tc, expected",
    argvalues=[(np.array([1.0]), np.array([23.0]), np.array([84.000000]))],
)
def test_calc_rnl(sf, tc, expected):
    """Tests calc_rnl.

    This test is intended to verify the implemented maths.
    """
    from pyrealm.core.solar import calc_rnl

    result = calc_rnl(sf, tc, CoreConst_fixture)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="k_alb_sw, tau, k_Gsc, dr, expected",
    argvalues=[
        (
            np.array([0.752844]),
            np.array([0.968381]),
            np.array([823.4242375]),
        )
    ],
)
def test_calc_rw(tau, dr, expected):
    """Test calc_rw.

    This test is intended to verify the implemented maths.
    """
    from pyrealm.core.solar import calc_rw

    result = calc_rw(tau, dr, CoreConst_fixture)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="rnl, tau, dr, delta, latitude, expected",
    argvalues=[
        (
            84.000000,
            np.array([0.752844]),
            np.array([0.968381]),
            np.array([23.436921]),
            np.array([37.7]),
            np.array([101.217016]),
        )
    ],
)
def test_calc_net_rad_crossover_hour_angle(rnl, tau, dr, delta, latitude, expected):
    """Tests calc_net_rad_crossover_hour_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calc_net_rad_crossover_hour_angle

    result = calc_net_rad_crossover_hour_angle(
        rnl, tau, dr, delta, latitude, CoreConst_fixture
    )

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


@pytest.mark.parametrize(
    argnames="latitude, declination, hour_angle, expected",
    argvalues=[
        (np.array([0.7854]), np.array([0.4093]), np.array([0]), np.array([1.19469633]))
    ],
)
def test_calc_beta_value(latitude, declination, hour_angle, expected):
    """Tests calc_beta_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calc_beta_angle_from_lat_dec_hour

    result = calc_beta_angle_from_lat_dec_hour(latitude, declination, hour_angle)

    assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        argnames="td, k_pir, expected",
        argvalues=[(np.array([298]), -0.23)],
    )
    def test_calc_declination(td, expected):
        """Tests calc_declination.

        This test is intended to verify the implemented maths.
        """

        from pyrealm.core.solar import test_calc_declination

        result = test_calc_declination(td, CoreConst_fixture)

        assert np.allclose(result, expected)
