import pytest


def test_profiling_example():
    from importlib import resources

    import numpy as np
    import xarray

    from pyrealm.pmodel import CalcCarbonIsotopes, PModel, PModelEnvironment

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

    # Profiling the PModel submodule
    # Standard C3 PModel
    pmod_c3 = PModel(env=pm_env, kphio=1 / 8)
    pmod_c3.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmod_c3.summarize()

    # Standard C4 PModel
    pmod_c4 = PModel(env=pm_env, kphio=1 / 8, method_optchi="c4")
    pmod_c4.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmod_c4.summarize()

    # Profiling the isotopes submodule
    # Create some entirely constant atmospheric isotope ratios
    d13CO2 = np.full_like(pm_env.tc, fill_value=-8.4)
    D14CO2 = np.full_like(pm_env.tc, fill_value=19.2)

    # Calculate for the C3 model
    carb_c3 = CalcCarbonIsotopes(pmod_c3, d13CO2=d13CO2, D14CO2=D14CO2)
    carb_c3.summarize()

    # Calculate for the C4 model
    carb_c4 = CalcCarbonIsotopes(pmod_c4, d13CO2=d13CO2, D14CO2=D14CO2)
    carb_c4.summarize()
