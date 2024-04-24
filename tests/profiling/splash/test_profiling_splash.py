"""Runs a profiler on the splash model to identify runtime bottlenecks."""

import pytest


@pytest.mark.profiling
def test_profile_splash(splash_profile_data):
    """Run a splash analysis for profiling."""

    from pyrealm.constants import CoreConst
    from pyrealm.splash.splash import SplashModel

    # SPLASH v1 uses 15°C / 288.1 5 K in the standard atmosphere definition and uses the
    # Chen method for calculating water density.

    splash_core_constants = CoreConst(k_To=288.15, water_density_method="chen")

    # Extract the input data
    sf, tc, pn, elv, lat, dates = splash_profile_data

    # Create the model
    splash = SplashModel(
        lat=lat,
        elv=elv,
        dates=dates,
        sf=sf,
        tc=tc,
        pn=pn,
        core_const=splash_core_constants,
    )

    # Run the initial soil calculation and then the time series.
    init_soil_moisture = splash.estimate_initial_soil_moisture(verbose=False)
    aet_out, wn_out, ro_out = splash.calculate_soil_moisture(init_soil_moisture)
