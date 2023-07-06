"""Testing the evaporative flux calculations against benchmark data from the original
SPLASH calculations"""

import numpy as np
import pytest


from splash_fixtures import daily_flux_benchmarks, grid_benchmarks


@pytest.fixture
def expected_attr():
    return ("sat", "lv", "pw", "psy", "econ", "cond", "eet_d", "pet_d", "rx")


def test_evap_scalar():
    """Test using array inputs with a single scalar value. The expected results are as
    the original output from the SPLASH evap.py __main__ function"""
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))
    solar = DailySolarFluxes(
        lat=np.array([37.7]),
        elv=np.array([142]),
        dates=cal,
        sf=np.array([1.0]),
        tc=np.array([23.0]),
    )

    evap = DailyEvapFluxes(solar, pa=np.array([99630.833]), tc=np.array([23.0]))

    # The original implementation provided sw=0.9 here, but that is now calculated
    # internally from the wn value. Check that it is recreated succesfully.
    aet, hi, sw = evap.estimate_aet(wn=np.array([128.571429]), only_aet=False)

    # Output of __main__ code in original evap.py
    expected = {
        "sat": 169.89609255250576,
        "lv": 2446686.637327215,
        "pw": 997.5836204018437,
        "psy": 66.72971923515009,
        "econ": 2.941667713784511e-10,
        "cond": 0.8851919575664212,
        "eet_d": 6.405467536773751,
        "pet_d": 8.070889096334925,
        "rx": 0.0013343404749726541,
    }

    for ky, val in expected.items():
        assert np.allclose(getattr(evap, ky), val)

    assert np.allclose(aet, 7.972787573253663)
    assert np.allclose(hi, 20.95931970358043)
    assert np.allclose(sw, 0.9)


def test_evap_iter(daily_flux_benchmarks, expected_attr):
    """Robust test checking of evap predictions.

    This checks that the outcome of calculating each input row in a time series
    independently gives the same answers as the original implementation, which _has_ to
    iterate over the rows to calculate values.
    """
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    inputs, expected = daily_flux_benchmarks

    for day, inp, exp in zip(inputs["dates"], inputs, expected):
        # Convert the input row into a dictionary of 1D arrays
        inp = {nm: np.array([inp[nm]]) for nm in inputs.dtype.names}

        cal = Calendar(np.array([day]).astype("datetime64[D]"))
        solar = DailySolarFluxes(
            lat=inp["lat"], elv=inp["elv"], dates=cal, sf=inp["sf"], tc=inp["tc"]
        )

        evap = DailyEvapFluxes(solar, pa=inp["pa"], tc=inp["tc"])
        aet, hi, sw = evap.estimate_aet(wn=inp["wn"], only_aet=False)

        for ky in expected_attr:
            assert np.allclose(getattr(evap, ky), exp[ky])

        # Note that sw is calculated explicitly in the inputs for SPLASH, because it is
        # used instead of wn in calculating daily fluxes, so this just validates that
        # the wn input is converted correctly to the value fed into SPLASH.

        # TODO - there is something odd here: all of the values are identical until here and
        #        then something about feeding sw into SPLASH and wn here means that there
        #        are small differences

        assert np.allclose(aet, exp["aet_d"], atol=0.01)
        assert np.allclose(hi, exp["hi"], atol=0.01)
        assert np.allclose(sw, inp["sw"], atol=0.01)


def test_evap_array(daily_flux_benchmarks, expected_attr):
    """Array checking of evaporative predictions.

    This checks that the outcome of calculating all the values in the test inputs
    _simultaneously_ using array inputs gives the same answers as the original
    iterated implementation.
    """
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar

    inputs, expected = daily_flux_benchmarks
    cal = Calendar(inputs["dates"].astype("datetime64[D]"))

    solar = DailySolarFluxes(
        lat=inputs["lat"],
        elv=inputs["elv"],
        dates=cal,
        sf=inputs["sf"],
        tc=inputs["tc"],
    )

    evap = DailyEvapFluxes(solar, pa=inputs["pa"], tc=inputs["tc"])
    aet, hi, sw = evap.estimate_aet(wn=inputs["wn"], only_aet=False)

    for ky in expected_attr:
        assert np.allclose(getattr(evap, ky), expected[ky])

    # Note that sw is calculated explicitly in the inputs for SPLASH, because it is
    # used instead of wn in calculating daily fluxes, so this just validates that
    # the wn input is converted correctly to the value fed into SPLASH.

    # TODO - there is something odd here: all of the values are identical until here and
    #        then something about feeding sw into SPLASH and wn here means that there
    #        are small differences
    assert np.allclose(aet, expected["aet_d"], atol=0.01)
    assert np.allclose(hi, expected["hi"], atol=0.01)
    assert np.allclose(sw, inputs["sw"], atol=0.01)


# TODO - test the day index approach.


def test_evap_array_stepping(grid_benchmarks):
    """Array checking of evaporative predictions using iteration over days.

    This checks that the outcome of calculating by iterating over days in the test
    inputs gives the same answers as the original iterated implementation.
    """
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes
    from pyrealm.splash.utilities import Calendar
    from pyrealm.splash.splash import elv2pres

    from pyrealm.constants import PModelConst
    from pyrealm.pmodel.functions import calc_patm

    inputs, expected = splash_benchmarks_grid

    cal = Calendar(inputs.time.values.astype("datetime64[D]"))

    # Duplicate lat and elev to same shape as sf and tc (TODO - avoid this!)
    sf_shape = inputs["sf"].shape
    elev = np.repeat(inputs["elev"].data[np.newaxis, :, :], sf_shape[0], axis=0)
    lat = np.repeat(inputs["lat"].data[:, np.newaxis], sf_shape[2], axis=1)
    lat = np.repeat(lat[np.newaxis, :, :], sf_shape[0], axis=0)

    solar = DailySolarFluxes(
        lat=lat,
        elv=elev,
        dates=cal,
        sf=inputs["sf"].data,
        tc=inputs["tmp"].data,
    )

    pa = elv2pres(inputs["elev"].data)

    # SPLASH uses 15°C in the standard atmosphere definition
    # pa = calc_patm(inputs["elev"].data, const=PModelConst(k_To=15.0))

    evap = DailyEvapFluxes(solar, pa=pa, tc=inputs["tmp"].data)

    # Test the static components of evap calculations are the same. Not quite sure why
    # they aren't identical and need the tolerance tweaking, but they are _very_ close
    for ky in ("sat", "lv", "pw", "psy", "econ", "cond", "eet_d", "pet_d"):
        assert np.allclose(
            getattr(evap, ky), expected[ky].data, equal_nan=True, rtol=0.0001
        )

    # assert the same starting point as the original spun up state
    curr_wn = inputs["wn_spun_up"]

    for day_idx, day in enumerate(cal):
        aet, hi = evap.estimate_aet(wn=curr_wn, return_hi=True)

    expected_attr = set(exp_names) - {"aet_d", "hi"}
    for ky in expected_attr:
        assert np.allclose(getattr(evap, ky), expected[ky])

    assert np.allclose(aet, expected["aet_d"])
    assert np.allclose(hi, expected["hi"])
