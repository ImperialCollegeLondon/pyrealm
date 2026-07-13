"""Regression test of SPLASH submodule.

Testing the evaporative flux calculations against benchmark data from the original
SPLASH calculations.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def expected_attr():
    """Define pyrealm names, original splash names and single test values.

    The values are the output of running __main__ in the original evap.py and provide a
    scalar test of the calculations.
    """

    return [
        ("saturation_slope", "sat", 169.89609255250576),
        ("enthalpy_vaporisation", "lv", 2446686.637327215),
        ("water_density", "pw", 997.5836204018437),
        ("psychrometric_constant", "psy", 66.72971923515009),
        ("water_energy_conversion", "econ", 2.941667713784511e-10),
        ("condensation", "cond", 0.8851919575664212),
        ("daily_eet", "eet_d", 6.405467536773751),
        ("daily_pet", "pet_d", 8.070889096334925),
        ("rx", "rx", 0.0013343404749726541),
    ]


def test_evap_scalar(splash_core_constants, expected_attr):
    """Test using array inputs with a single scalar value.

    The expected results are as the original output from the SPLASH evap.py __main__
    function.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.evap import DailyEvaporativeFluxes
    from pyrealm.splash.solar import DailySolarFluxes

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))
    solar = DailySolarFluxes(
        latitude=np.array([37.7]),
        elevation=np.array([142]),
        dates=cal,
        sunshine_fraction=np.array([1.0]),
        temperature=np.array([23.0]),
    )

    evap = DailyEvaporativeFluxes(
        solar,
        patm=np.array([99630.833]),
        temperature=np.array([23.0]),
        core_const=splash_core_constants,
    )

    for ky, _, val in expected_attr:
        assert_allclose(getattr(evap, ky), val)

    # The original implementation provided sw=0.9 here, but that is now calculated
    # internally from the wn value. Check that it is recreated successfully.
    aet, hi, sw = evap.estimate_aet(
        soil_moisture=np.array([128.571429]), only_aet=False
    )

    assert_allclose(aet, 7.972787573253663)
    assert_allclose(hi, 20.95931970358043)
    assert_allclose(sw, 0.9)


def test_evap_iter(splash_core_constants, daily_flux_benchmarks, expected_attr):
    """Robust test checking of evap predictions.

    This checks that the outcome of calculating each input row in a time series
    independently gives the same answers as the original implementation, which _has_ to
    iterate over the rows to calculate values.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.evap import DailyEvaporativeFluxes
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = daily_flux_benchmarks

    for day, (_, inp), (_, exp) in zip(
        inputs["dates"], inputs.iterrows(), expected.iterrows()
    ):
        cal = Calendar(dates=np.array([day]).astype("datetime64[D]"))
        solar = DailySolarFluxes(
            dates=cal,
            latitude=np.array([inp["lat"]]),
            elevation=np.array([inp["elv"]]),
            sunshine_fraction=np.array([inp["sf"]]),
            temperature=np.array([inp["tc"]]),
        )

        evap = DailyEvaporativeFluxes(
            solar=solar,
            patm=np.array([inp["pa"]]),
            temperature=np.array([inp["tc"]]),
            core_const=splash_core_constants,
        )
        aet, hi, _ = evap.estimate_aet(soil_moisture=inp["wn"], only_aet=False)

        for pyrealm_attr, splash_key, _ in expected_attr:
            assert_allclose(getattr(evap, pyrealm_attr), exp[splash_key])

        # Check the values returned by estimate_aet
        assert_allclose(aet, exp["aet_d"])
        assert_allclose(hi, exp["hi"])


def test_evap_array(splash_core_constants, daily_flux_benchmarks, expected_attr):
    """Array checking of evaporative predictions.

    This checks that the outcome of calculating all the values in the test inputs
    _simultaneously_ using array inputs gives the same answers as the original
    iterated implementation.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.evap import DailyEvaporativeFluxes
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = daily_flux_benchmarks
    cal = Calendar(inputs["dates"].to_numpy().astype("datetime64[D]"))

    solar = DailySolarFluxes(
        dates=cal,
        latitude=inputs["lat"].to_numpy(),
        elevation=inputs["elv"].to_numpy(),
        sunshine_fraction=inputs["sf"].to_numpy(),
        temperature=inputs["tc"].to_numpy(),
    )

    evap = DailyEvaporativeFluxes(
        solar=solar,
        patm=inputs["pa"].to_numpy(),
        temperature=inputs["tc"].to_numpy(),
        core_const=splash_core_constants,
    )
    aet, hi, _ = evap.estimate_aet(
        soil_moisture=inputs["wn"].to_numpy(), only_aet=False
    )

    for pyrealm_attr, splash_key, _ in expected_attr:
        assert_allclose(getattr(evap, pyrealm_attr), expected[splash_key])

    # Check the values returned by estimate_aet
    assert_allclose(aet, expected["aet_d"])
    assert_allclose(hi, expected["hi"])


def test_evap_array_grid(splash_core_constants, grid_benchmarks, expected_attr):
    """Array checking of evaporative predictions using iteration over days.

    This checks that the outcome of evaporative calculations from running the full
    SPLASH model on a gridded dataset are consistent.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.core.pressure import calculate_patm
    from pyrealm.splash.evap import DailyEvaporativeFluxes
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = grid_benchmarks

    cal = Calendar(inputs.time.values.astype("datetime64[D]"))

    # Make lat and elev broadcastable to sf and tc
    lat = inputs.lat.to_numpy()[None, :, None]
    elev = inputs.elev.to_numpy()[None, :, :]

    solar = DailySolarFluxes(
        latitude=lat,
        elevation=elev,
        dates=cal,
        sunshine_fraction=inputs["sf"].to_numpy(),
        temperature=inputs["tmp"].to_numpy(),
        core_const=splash_core_constants,
    )

    pa = calculate_patm(elev, core_const=splash_core_constants)

    evap = DailyEvaporativeFluxes(
        solar,
        patm=pa,
        temperature=inputs["tmp"].data,
        core_const=splash_core_constants,
    )

    # Test the static components of evap calculations are the same - which can be
    # tested across the whole array
    for pyrealm_attr, splash_key, _ in expected_attr:
        if pyrealm_attr == "rx":
            continue

        assert_allclose(
            getattr(evap, pyrealm_attr),
            expected[splash_key].to_numpy(),
            equal_nan=True,
            rtol=1e-5,
        )

    # Now validate the expected AET - because the whole soil moisture sequence has
    # been created in the original implementation, the whole time sequence can be passed
    # in as a single array and calculated without daily iteration, *but* the soil
    # moisture used to calculate AET is from the preceding day, so need to shift the wn
    # sequence to start with the spun up values and drop the last day.
    wn_spun_up = np.expand_dims(expected["wn_spun_up"].to_numpy(), axis=0)
    wn_sequence = np.vstack([wn_spun_up, expected["wn"].to_numpy()[:-1, :, :]])

    aet = evap.estimate_aet(soil_moisture=wn_sequence, day_index=None)
    assert_allclose(aet, expected["aet_d"], equal_nan=True, rtol=2e-6)
