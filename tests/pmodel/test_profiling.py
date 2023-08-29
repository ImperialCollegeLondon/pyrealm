import pytest


def test_profiling_example():
    from importlib import resources

    import numpy as np
    import xarray

    from pyrealm.pmodel import PModel, PModelEnvironment

    # Loading the dataset:
    dpath = resources.files("pyrealm_build_data") / "inputs_data_24.25.nc"

    ds = xarray.load_dataset(dpath)

    # Variable set up
    # Air temperature in Kelvin
    tc = (ds["temp"]).to_numpy()
    # Atmospheric pressure in Pascals
    patm = ds["patm"].to_numpy()
    # Obtain VPD and remove negative values
    vpd = ds["vpd"].to_numpy()
    vpd = np.clip(vpd, 0, np.inf)
    # Extract fAPAR (unitless)
    fapar = ds["fapar"].to_numpy()
    # Gather PPFD µmole/m2/s1
    ppfd = ds["ppfd"].to_numpy()
    # Define atmospheric CO2 concentration (ppm)
    co2 = np.ones_like(tc) * 400

    # Generate and check the PModelEnvironment
    pm_env = PModelEnvironment(tc=tc, patm=patm, vpd=vpd, co2=co2)

    # Standard C3 PModel
    pmod_c3 = PModel(env=pm_env, kphio=1 / 8)
    pmod_c3.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmod_c3.summarize()

    # Standard C4 PModel
    pmod_c4 = PModel(env=pm_env, kphio=1 / 8, method_optchi="c4")
    pmod_c4.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmod_c4.summarize()
