"""This test module tests that the calculation of solar fluxes from shortwave radiation
(as opposed to the original calculation from sunshine fraction) matches the
implementation used in rsplash (SPLASH v2).
"""  # noqa: D205

from importlib import resources

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def bourne():
    """Import the Bourne site dataset inputs and rsplash predictions."""

    dpath = resources.files("pyrealm_build_data.rsplash")

    inputs = pd.read_csv(dpath / "rsplash_Bourne_inputs.csv")
    internals = pd.read_csv(dpath / "rsplash_Bourne_internals.csv")
    expected = pd.read_csv(dpath / "rsplash_Bourne_outputs.csv")

    return inputs, internals, expected


def test_splash_with_swdown(bourne):
    """Test using shortwave inputs with SplashModel against rsplash example."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, internals, outputs = bourne

    calendar = Calendar(inputs["date"].to_numpy().astype("datetime64[s]"))

    # SPLASH v2 has variable albedo with snowpack and has slope and other factors.
    # Rather than figuring out conversion factors to align responses, force SplashModel
    # to use the calculated albedo values from SPLASH 2
    const = CoreConst()
    object.__setattr__(const, "shortwave_albedo", internals["alb"].to_numpy())

    # Run the model
    model = SplashModel(
        lat=inputs["lat"].to_numpy(),
        elv=inputs["elev"].to_numpy(),
        tc=inputs["Ta"].to_numpy(),
        pn=inputs["P"].to_numpy(),
        shortwave_radiation=inputs["sw_in"].to_numpy(),
        dates=calendar,
        kWm=np.array([250]),
        core_const=const,
    )

    model.calculate_soil_moisture(wn_init=outputs["wn"][0])

    assert_allclose(model.solar.nu, internals["nu"].to_numpy(), atol=1e-6)
    assert_allclose(model.solar.lambda_, internals["lambda"].to_numpy(), atol=1e-6)
    assert_allclose(model.solar.distance_factor, internals["dr"].to_numpy(), atol=1e-6)
    assert_allclose(model.solar.declination, internals["delta"].to_numpy(), atol=1e-6)
    assert_allclose(model.solar.ru, internals["ru"].to_numpy(), atol=1e-6)
    assert_allclose(model.solar.rv, internals["rv"].to_numpy(), atol=1e-6)
    assert_allclose(
        model.solar.sunset_hour_angle, internals["hs"].to_numpy(), atol=1e-6
    )
    assert_allclose(
        model.solar.clear_sky_transmissivity, internals["tau_o"].to_numpy(), atol=1e-6
    )
    assert_allclose(
        model.solar.daily_solar_radiation / 1e6, internals["Ho"].to_numpy(), atol=1e-6
    )
    assert_allclose(model.solar.transmissivity, internals["tau"].to_numpy(), atol=1e-6)
    assert_allclose(model.solar.daily_ppfd, internals["PPFD"].to_numpy(), atol=1e-6)
    assert_allclose(
        model.solar.sunshine_fraction, internals["sf"].to_numpy(), atol=1e-6
    )
    assert_allclose(
        model.solar.net_longwave_radiation, internals["Rnl"].to_numpy(), atol=1e-6
    )

    # These tests use a different tolerance because of the reduced precision of the
    # albedo values being inserted into the calculations
    assert_allclose(model.solar.rw, internals["rw"].to_numpy(), rtol=1e-5)

    assert_allclose(
        model.solar.crossover_hour_angle,
        internals["hn"].to_numpy(),
        rtol=1e-5,
    )

    assert_allclose(
        model.solar.daytime_net_radiation,
        internals["Hn"].to_numpy() * 1e6,
        rtol=1e-5,
    )

    assert_allclose(
        model.solar.nighttime_net_radiation,
        internals["Hnn"].to_numpy() * 1e6,
        rtol=1e-5,
    )
