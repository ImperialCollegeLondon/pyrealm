import pytest


def test_profiling_example():
    from importlib import resources

    import numpy as np
    import xarray

    from pyrealm.pmodel import (
        C3C4Competition,
        CalcCarbonIsotopes,
        PModel,
        PModelEnvironment,
    )

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

    # Profiling the Competition submodule
    # Competition, using annual GPP from µgC m2 s to g m2 yr
    gpp_c3_annual = pmod_c3.gpp * (60 * 60 * 24 * 365) * 1e-6
    gpp_c4_annual = pmod_c4.gpp * (60 * 60 * 24 * 365) * 1e-6

    # Fit the competition model - making some extrenely poor judgements about what is
    # cropland and what is below the minimum temperature that really should be fixed.
    comp = C3C4Competition(
        gpp_c3=gpp_c3_annual,
        gpp_c4=gpp_c4_annual,
        treecover=np.array([0.5]),
        below_t_min=np.array([False]),
        cropland=np.array([False]),
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

    # Calculate the expected isotopic patterns in locations given the competition model
    comp.estimate_isotopic_discrimination(
        d13CO2=constant_d13CO2,
        Delta13C_C3_alone=isotope_c3.Delta13C,
        Delta13C_C4_alone=isotope_c4.Delta13C,
    )

    comp.summarize()
