"""This module tests the main methods in the SplashModel.

- estimate_daily_water_balance, which estimates soil moisture and run off given
  preceding soil moisture
- estimate_initial_soil_moisture, which assumes a stationary relationship over an
  annual cycle to estimate a starting soil moisture.
- calculate_soil_moisture, which iterates an initial soil moisture forward over a time
  series.
"""

import numpy as np
from numpy.testing import assert_allclose

# Testing estimate_daily_water_balance (was run_one_day)


def test_estimate_daily_water_balance_scalar(splash_core_constants):
    """Tests a single day calculation.

    Uses the expectations from the __main__ example provided in SPLASH v1.0 splash.py.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))
    splash = SplashModel(
        latitude=np.array([37.7]),
        elevation=np.array([142]),
        sunshine_fraction=np.array([1.0]),
        temperature=np.array([23.0]),
        precipitation=np.array([5]),
        dates=cal,
        core_const=splash_core_constants,
    )
    aet, sm, ro = splash.estimate_daily_water_balance(
        previous_soil_moisture=np.array(75), day_index=0
    )

    # Expected values are the output of __main__ in original splash.py
    evap_expected = {
        "condensation": np.array([0.885192]),
        "daily_eet": np.array([6.405468]),
        "daily_pet": np.array([8.070889]),
    }

    for ky, val in evap_expected.items():
        assert_allclose(getattr(splash.evap, ky), val)

    assert_allclose(aet, 5.748034)
    assert_allclose(sm, 75.137158)
    assert_allclose(ro, 0.0000000)


def test_estimate_daily_water_balance_iter(
    splash_core_constants, daily_flux_benchmarks
):
    """Test iterated water balance.

    This test iterates over the individual daily benchmark rows, calculating each
    prediction as a single independent day.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = daily_flux_benchmarks
    days = inputs["dates"].to_numpy().astype("datetime64[D]")

    for day, (_, inp), (_, exp) in zip(days, inputs.iterrows(), expected.iterrows()):
        # initialise splash and calculate the evaporative flux and soil moisture
        splash = SplashModel(
            latitude=np.array([inp["lat"]]),
            elevation=np.array([inp["elv"]]),
            sunshine_fraction=np.array([inp["sf"]]),
            temperature=np.array([inp["tc"]]),
            precipitation=np.array([inp["pn"]]),
            dates=Calendar(np.array([day])),
            core_const=splash_core_constants,
        )
        aet, sm, ro = splash.estimate_daily_water_balance(
            previous_soil_moisture=np.array([inp["wn"]]), day_index=0
        )
        assert_allclose(aet, exp["aet_d"], rtol=1e-6)
        assert_allclose(sm, exp["wn"], rtol=1e-6)
        assert_allclose(ro, exp["ro"], rtol=1e-6)


def test_estimate_daily_water_balance_array(
    splash_core_constants, daily_flux_benchmarks
):
    """This test runs the individual daily benchmark data as an array."""
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = daily_flux_benchmarks

    splash = SplashModel(
        latitude=inputs["lat"].to_numpy(),
        elevation=inputs["elv"].to_numpy(),
        sunshine_fraction=inputs["sf"].to_numpy(),
        temperature=inputs["tc"].to_numpy(),
        precipitation=inputs["pn"].to_numpy(),
        dates=Calendar(inputs["dates"].to_numpy()),
        core_const=splash_core_constants,
    )

    aet, sm, ro = splash.estimate_daily_water_balance(
        previous_soil_moisture=inputs["wn"].to_numpy(), day_index=None
    )

    assert_allclose(aet, expected["aet_d"], rtol=1e-6)
    assert_allclose(sm, expected["wn"], rtol=1e-6)
    assert_allclose(ro, expected["ro"], rtol=1e-6)


# Testing the spin-up process


def test_run_spin_up_oned(splash_core_constants, one_d_benchmark):
    """Test the spin up process using the original 1D test data from __main__.py."""
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = one_d_benchmark

    # Need to reshape the inputs so they have a time and 1 observation axis and
    # duplicate lat and elev to same shape as sf, tc, pc

    splash = SplashModel(
        latitude=inputs.lat.to_numpy()[None, :, None],
        elevation=inputs.elev.to_numpy()[None, :, :],
        sunshine_fraction=inputs.sf.to_numpy(),
        temperature=inputs.tmp.to_numpy(),
        precipitation=inputs.pre.to_numpy(),
        dates=Calendar(inputs.time.to_numpy()),
        core_const=splash_core_constants,
    )

    wn = splash.estimate_initial_soil_moisture()

    # Check against the spun up value from the original implementation
    assert_allclose(wn, expected["wn_spun_up"])


def test_run_spin_up_gridded(splash_core_constants, grid_benchmarks):
    """Test the spin up process using the grid in a single pass across observations."""

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = grid_benchmarks

    splash = SplashModel(
        latitude=inputs.lat.to_numpy()[None, :, None],
        elevation=inputs.elev.to_numpy()[None, :, :],
        sunshine_fraction=inputs.sf.to_numpy(),
        temperature=inputs.tmp.to_numpy(),
        precipitation=inputs.pre.to_numpy(),
        dates=Calendar(inputs.time.to_numpy()),
        core_const=splash_core_constants,
    )

    wn = splash.estimate_initial_soil_moisture()

    # Check against the spun up value from the original implementation
    expected_wn = expected["wn_spun_up"].to_numpy()
    assert_allclose(wn, expected_wn, equal_nan=True, rtol=1e-6)


# Testing the iterated water balance calculation


def test_calculate_soil_moisture_oned(splash_core_constants, one_d_benchmark):
    """Test the water balance iteration.

    Uses the original 1D test data from __main__.py.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = one_d_benchmark

    # Use the xarray dimensions to handle broadcasting.
    splash = splash = SplashModel(
        latitude=inputs.lat,
        elevation=inputs.elev,
        sunshine_fraction=inputs.sf,
        temperature=inputs.tmp,
        precipitation=inputs.pre,
        dates=Calendar(inputs.time),
        core_const=splash_core_constants,
    )

    # Start from the existing spun up start point in the SPLASH outputs - creation of
    # this input is tested above.
    aet, wn, ro = splash.calculate_soil_moisture(expected["wn_spun_up"].data)

    assert_allclose(splash.evap.daily_pet, expected["pet_d"].data)

    # Check against the spun up value from the original implementation
    assert_allclose(aet, expected["aet_d"].data, rtol=1e-6)
    assert_allclose(wn, expected["wn"].data, rtol=1e-6)
    assert_allclose(ro, expected["ro"].data, rtol=1e-6)


def test_calculate_soil_moisture_grid(splash_core_constants, grid_benchmarks):
    """Test the water balance iteration on a grid.

    Uses the original 1D test data from __main__.py.
    """

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = grid_benchmarks

    # Use the xarray dimensions to handle broadcasting.
    splash = SplashModel(
        latitude=inputs.lat,
        elevation=inputs.elev,
        sunshine_fraction=inputs.sf,
        temperature=inputs.tmp,
        precipitation=inputs.pre,
        dates=Calendar(inputs.time),
        core_const=splash_core_constants,
    )

    # Start from the existing spun up start point in the SPLASH outputs - creation of
    # this input is tested above.
    aet, wn, ro = splash.calculate_soil_moisture(expected["wn_spun_up"].data)

    # Check against the spun up value from the original implementation
    assert_allclose(aet, expected["aet_d"].data, equal_nan=True, rtol=2e-6)
    assert_allclose(wn, expected["wn"].data, equal_nan=True, rtol=2e-6)
    # Not entirely clear where the slight differences come from
    assert_allclose(ro, expected["ro"].data, equal_nan=True, rtol=2e-4)
