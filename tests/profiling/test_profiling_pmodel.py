"""Profiling test class.

Runs a profiler on the pmodel implementation to identify runtime
bottlenecks.
"""

from importlib import resources

import numpy as np
import pytest
import xarray

from pyrealm.pmodel.pmodel_environment import PModelEnvironment


@pytest.fixture(scope="module")
def setup():
    """Setting up the pmodel and loading the test data set.

    This uses a module scoped yield fixture to only load the data once and then to
    delete it all when the tests exit.
    """

    print("\n***\nSetting up profiling of pmodel\n***\n")

    # Loading the dataset:
    dpath = resources.files("pyrealm_build_data") / "inputs_data_reduced_1y.nc"

    ds = xarray.load_dataset(dpath)

    # TODO - this is a bit of a hack because of the current unfriendly handling of
    #  data that does not form neat blocks of daily data in the subdaily module. The
    #        test data is a slice along longitude 24.25°E (Finland --> Crete -->
    #        Botswana) so the actual local times needed for the subdaily module are
    #        offset from the UTC times in the data. This step reduces the input data
    #        to complete daily blocks of data using local time

    ds = ds.sel(time=slice("2000-01-01T01:59", "2000-12-31T01:59"))
    ds["local_time_offset"] = (ds["lon"] / 15 * 3600).astype("timedelta64[s]")
    # 15° per hour, 3600 seconds per hour
    ds["local_time"] = ds["time"] - ds["local_time_offset"]

    # Function to increase the size of the test data by broadcasting identical
    # values along the longitude axis and tiling year repeats
    years = 1

    def _scale_up(array, lon=40, years=2):
        # Broadcast along the last axis
        more_lons = np.broadcast_to(array, list(array.shape[:-1]) + [lon])
        # Tile along the time axis
        return np.tile(more_lons, (years, 1, 1))

    # Variable set up
    # Air temperature in Kelvin
    tc = _scale_up(ds["temp"].to_numpy(), years=years)
    # Atmospheric pressure in Pascals
    patm = _scale_up(ds["patm"].to_numpy(), years=years)
    # Obtain VPD and remove negative values
    vpd = ds["vpd"].to_numpy()
    vpd = _scale_up(np.clip(vpd, 0, np.inf), years=years)
    # Extract fAPAR (unitless)
    fapar = _scale_up(ds["fapar"].to_numpy(), years=years)
    # Gather PPFD µmole/m2/s1
    ppfd = _scale_up(ds["ppfd"].to_numpy(), years=years)
    # Define atmospheric CO2 concentration (ppm)
    co2 = _scale_up(ds["co2"].to_numpy(), years=years)

    # Expand the time to the number of years
    local_time = ds["local_time"].to_numpy().squeeze()
    local_time = [local_time] * years
    for yr in range(years):
        local_time[yr] = local_time[yr] + np.timedelta64(yr, "Y").astype(
            "timedelta64[ns]"
        )
    local_time = np.concatenate(local_time)

    # Generate and check the PModelEnvironment
    pm_env = PModelEnvironment(tc=tc, patm=patm, vpd=vpd, co2=co2)

    yield pm_env, fapar, ppfd, local_time

    print("\n***\nTearing down\n***\n")

    # Delete all the local objects
    del list(locals().keys())[:]


@pytest.mark.profiling
def test_profiling_example(setup):
    """Running the profiler on the pmodel."""
    from pyrealm.pmodel import C3C4Competition, CalcCarbonIsotopes, PModel

    # Unpack feature components
    pm_env, fapar, ppfd, local_time = setup

    # Profiling the PModel submodule
    # Standard C3 PModel
    pmod_c3 = PModel(env=pm_env, kphio=1 / 8)
    pmod_c3.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmod_c3.summarize()

    # Standard C4 PModel
    pmod_c4 = PModel(env=pm_env, kphio=1 / 8, method_optchi="c4")
    pmod_c4.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmod_c4.summarize()

    # Profiling the Competition submodule
    # Competition, using annual GPP from µgC m2 s to g m2 yr
    gpp_c3_annual = pmod_c3.gpp * (60 * 60 * 24 * 365) * 1e-6
    gpp_c4_annual = pmod_c4.gpp * (60 * 60 * 24 * 365) * 1e-6

    # Fit the competition model - making some extrenely poor judgements about what
    # is cropland and what is below the minimum temperature that really should be
    # fixed.
    comp = C3C4Competition(
        gpp_c3=gpp_c3_annual,
        gpp_c4=gpp_c4_annual,
        treecover=np.array([0.5]),
        below_t_min=np.full_like(pm_env.tc, False, dtype="bool"),
        cropland=np.full_like(pm_env.tc, False, dtype="bool"),
    )

    comp.summarize()

    # Profiling the isotopes submodule
    # Create some entirely constant atmospheric isotope ratios
    constant_d13CO2 = np.array([-8.4])
    constant_D14CO2 = np.array([19.2])

    # Calculate for the C3 model
    isotope_c3 = CalcCarbonIsotopes(
        pmod_c3, d13CO2=constant_d13CO2, D14CO2=constant_D14CO2
    )
    isotope_c3.summarize()

    # Calculate for the C4 model
    isotope_c4 = CalcCarbonIsotopes(
        pmod_c4, d13CO2=constant_d13CO2, D14CO2=constant_D14CO2
    )
    isotope_c4.summarize()

    # Calculate the expected isotopic patterns in locations given the competition
    # model
    comp.estimate_isotopic_discrimination(
        d13CO2=constant_d13CO2,
        Delta13C_C3_alone=isotope_c3.Delta13C,
        Delta13C_C4_alone=isotope_c4.Delta13C,
    )

    comp.summarize()


@pytest.mark.profiling
def test_profiling_subdaily(setup):
    """Profiling the subdaily submodule."""
    from pyrealm.pmodel import SubdailyPModel, SubdailyScaler

    # Unpack feature components
    pm_env, fapar, ppfd, local_time = setup

    # SubdailyPModel with 1 hour noon acclimation window
    fsscaler = SubdailyScaler(local_time)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(1, "h"),
    )
    subdaily_pmod = SubdailyPModel(
        env=pm_env,
        fs_scaler=fsscaler,
        allow_holdover=True,
        fapar=fapar,
        ppfd=ppfd,
        alpha=1 / 15,
    )
    return subdaily_pmod
