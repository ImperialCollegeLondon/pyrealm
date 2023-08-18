import pytest

def test_profiling_example():

    from importlib import resources
    import xarray
    import numpy as np
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

    # Standard PModel
    pmod = PModel(env=pm_env, kphio=1 / 8)
    pmod.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmod.summarize()