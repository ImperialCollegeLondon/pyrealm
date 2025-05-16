"""Exemplar calculation of LAI predictions using Boya Zhou's method.

This is a script implementation of the calculation of a phenological LAI time series
based on Boya Zhou's approach. The original paper uses a mix of different inputs and
coding languages - this script was put together by David Orme and Boya Zhou to bring
all of the calculations into Python using agreed inputs to create a repeatable
regression test dataset.

The outputs include calculations from:
* a half hourly subdaily P model using soil moisture stress
* a fortnightly PModel with no soil moisture stress
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy.special import lambertw  # type: ignore[import-untyped]

from pyrealm.core.time_series import AnnualValueCalculator
from pyrealm.core.water import convert_water_mm_to_moles
from pyrealm.pmodel import (
    AcclimationModel,
    PModel,
    PModelEnvironment,
    SubdailyPModel,
)
from pyrealm.pmodel.functions import calc_soilmstress_mengoli


# Local RLE function - will be in pyrealm.demography.growing_season
def run_length_encode(values: NDArray) -> tuple[NDArray[np.int_], NDArray]:
    """Calculate run length encoding of 1D arrays.

    The function returns a tuple containing an array of the run lengths and an array of
    the values of each run. These can be turned back into the original array using
    ``np.repeat(values, run_lengths)``.

    Args:
        values: A one dimensional array of values
    """

    n = values.size
    if n == 0 or values.ndim != 1:
        raise ValueError(
            "run_length_encode requires a 1 dimensional array of non-zero length"
        )

    # Find where adjacent values are not equal
    pairs_not_equal = values[1:] != values[:-1]
    # Find change points where values are not equal and add last position.
    change_points = np.append(np.where(pairs_not_equal), n - 1)
    # Get run lengths between change points
    run_lengths = np.diff(np.append(-1, change_points))

    return (run_lengths, values[change_points])


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------

# Load the site data
with open("DE-GRI_site_data.json") as dpath:
    de_gri_site_data = json.load(dpath)

# Load the half hourly data - ignoring mypy's dislike of perfectly functional numeric
# inputs to na_values.
de_gri_hh_pd = pd.read_csv(
    "DE_GRI_hh_fluxnet_simple.csv",
    na_values=["-9999-9999.0", -9999.0, -9999],  # type: ignore[list-item]
)

# Calculate time as np.datetime64, set as the index and convert to xarray
de_gri_hh_pd["time"] = pd.to_datetime(
    de_gri_hh_pd["TIMESTAMP_START"], format="%Y%m%d%H%M"
)
de_gri_hh_pd = de_gri_hh_pd.set_index("time")
de_gri_hh_xr = de_gri_hh_pd.to_xarray()


# # Blank out temperatures under 25°C
de_gri_hh_xr["TA_F"] = de_gri_hh_xr["TA_F"].where(de_gri_hh_xr["TA_F"] >= -25)

# # VPD from hPa to Pa
de_gri_hh_xr["VPD_F"] = de_gri_hh_xr["VPD_F"] * 100
# Pressure from kPa to Pa
de_gri_hh_xr["PA_F"] = de_gri_hh_xr["PA_F"] * 1000
# PPFD from SWDOWN
de_gri_hh_xr["PPFD"] = de_gri_hh_xr["SW_IN_F_MDS"] * 2.04

# Convert precipitation to molar values at half hour scale to aggregate up to annual
# totals. Can't simply convert annual means here - need to convert with conditions at
# half hourly time step.
#
# - Both FluxNET and CRU (loaded below for aridity and soil moisture calculations)
#   provide precipitation data. CRU is more consistent with the aridity index
#   calculation and hence f_0, but the FluxNET data is more site appropriate so is used
#   here. We also need Temp and PATM to convert water mm to water mols, and currently
#   the soil moisture inputs don't include that from the daily CRU data.

# Calculate water as mols m2 not mm m2
site_precip_molar = convert_water_mm_to_moles(
    water_mm=de_gri_hh_xr["P_F"].to_numpy(),
    tc=de_gri_hh_xr["TA_F"].to_numpy(),
    patm=de_gri_hh_xr["PA_F"].to_numpy(),
)

de_gri_hh_xr = de_gri_hh_xr.assign(P_F_MOLAR=("time", site_precip_molar))

# Calculate daily soil moisture stress penalties

# 1. Load Soil moisture from SPLASH run on CRU TS 4.07
de_gri_splash = xr.load_dataset("DE_gri_splash_cru_ts4.07_2000_2019.nc")

# 2. Calculate 20 year aridity index (2000 - 2020) as PET/P
aridity_index = de_gri_splash["pet"].mean() / de_gri_splash["pre"].mean()

# 3. Store that in the site data
with open("DE-GRI_site_data.json", "w") as dpath:
    de_gri_site_data["AI_from_cruts"] = float(aridity_index)
    json.dump(obj=de_gri_site_data, fp=dpath)

# --------------------------------------------------------------------------------------
# Subdaily model with Mengoli soil moisture stress
# --------------------------------------------------------------------------------------

# Calculate the PModel photosynthetic environment, setting fapar = 1 to estimate
# potential GPP

env = PModelEnvironment(
    tc=de_gri_hh_xr["TA_F"].to_numpy(),
    vpd=de_gri_hh_xr["VPD_F"].to_numpy(),
    co2=de_gri_hh_xr["CO2_F_MDS"].to_numpy(),
    patm=de_gri_hh_xr["PA_F"].to_numpy(),
    fapar=np.ones_like(de_gri_hh_xr["TA_F"]),
    ppfd=de_gri_hh_xr["PPFD"].to_numpy(),
)

# Set up the datetimes of the observations and set the acclimation window
acclim = AcclimationModel(
    datetimes=de_gri_hh_xr["time"].to_numpy(), alpha=1 / 15, allow_holdover=True
)
acclim.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(30, "m"),
)

# Fit the subdaily potential GPP: fAPAR = 1 as set above and phi0 = 1/8
subdaily_pmodel = SubdailyPModel(
    env=env,
    acclim_model=acclim,
    reference_kphio=1 / 8,
    method_kphio="temperature",
)

# Get an xarray dataset of the required outputs at half hourly scale
subdaily_outputs = xr.Dataset(
    data_vars=dict(
        PMod_gpp=("time", subdaily_pmodel.gpp),
        PMod_chi=("time", subdaily_pmodel.optchi.chi),
        PMod_ci=("time", subdaily_pmodel.optchi.ci),
        ca=("time", env.ca),
        tc=("time", env.tc),
        co2=("time", env.co2),
        vpd=("time", env.vpd),
        patm=("time", env.patm),
        ppfd=("time", env.ppfd),
        precip_molar=de_gri_hh_xr["P_F_MOLAR"],
    ),
    coords=dict(time=de_gri_hh_xr["time"]),
)

# To calculate the fAPAR and LAI max, we need a definition of growing season. We don't
# have good methods for this yet, but the example in the demo code was any day with a
# mean temperature of greater than freezing, and where that continued for at least 5
# days.

# Get the growing days
growing_day = subdaily_outputs["tc"].resample(time="1D").mean() > 0

# Eliminate short chunks of growth
gsl_lengths, gsl_values = run_length_encode(growing_day.to_numpy())
gsl_values[np.logical_and(gsl_values == 1, gsl_lengths < 5)] = 0
growing_day.data = np.repeat(gsl_values, gsl_lengths)

# Now assign those daily values to the subdaily observations
subdaily_outputs = subdaily_outputs.assign(
    growing_day=("time", np.repeat(growing_day.to_numpy(), 48))
)

# We also want to support posthoc GPP penalties, so calculate the Mengoli daily stress
# factors using SPLASH soil moisture / bucket size
de_gri_splash = de_gri_splash.assign(
    soilm_stress=(
        "time",
        calc_soilmstress_mengoli(
            soilm=de_gri_splash["wn"].to_numpy() / 150,
            aridity_index=aridity_index.to_numpy(),
        ),
    )
)

# Reduce to GPP timeseries length
de_gri_splash = de_gri_splash.sel(time=slice("2004-01-01", "2014-12-31"))

# Duplicate to half hourly intervals in subdaily data and calculate penalised GPP
subdaily_outputs = subdaily_outputs.assign(
    soilm_stress=("time", np.repeat(de_gri_splash["soilm_stress"].to_numpy(), 48))
)

subdaily_outputs = subdaily_outputs.assign(
    PMod_gpp_smstress=subdaily_outputs["PMod_gpp"] * subdaily_outputs["soilm_stress"],
)


# Calculate annual values
avc = AnnualValueCalculator(
    timing=acclim,
    growing_season=subdaily_outputs["growing_day"].to_numpy(),
)


# Calculate actual assimilation during the observation
subdaily_outputs = subdaily_outputs.assign(
    PMod_gpp_smstress=subdaily_outputs["PMod_gpp"] * subdaily_outputs["soilm_stress"],
)


# Average annual GPP ± soil moisture stress
# GPP, precipitation and growing day totals across the whole year.

ann_mean_subdaily_gpp = avc.get_annual_means(subdaily_outputs["PMod_gpp"].to_numpy())

ann_mean_subdaily_gpp_smstress = avc.get_annual_means(
    subdaily_outputs["PMod_gpp_smstress"].to_numpy()
)

ann_total_P_molar = avc.get_annual_totals(subdaily_outputs["precip_molar"].to_numpy())

# This is awkward - need to extract these from the AVC somehow - need to go from AVC to
# number of days and number of growing days.

n_days = de_gri_splash["time"].resample(time="1YE").count()

ann_total_GD = (
    avc.get_annual_totals(subdaily_outputs["growing_day"].to_numpy()) / 48
).astype(np.int_)


# Chi, ca and VPD in growing season
annual_mean_ca_in_GS = avc.get_annual_means(
    subdaily_outputs["ca"].to_numpy(), within_growing_season=True
)

annual_mean_chi_in_GS = avc.get_annual_means(
    subdaily_outputs["PMod_chi"].to_numpy(), within_growing_season=True
)

annual_mean_vpd_in_GS = avc.get_annual_means(
    subdaily_outputs["vpd"].to_numpy(), within_growing_season=True
)

# Create an annual dataset, joining on site data to drop extra CRU years
subdaily_annual_values = xr.Dataset(
    data_vars=dict(
        ann_mean_subdaily_gpp_smstress=("time", ann_mean_subdaily_gpp_smstress),
        ann_mean_subdaily_gpp=("time", ann_mean_subdaily_gpp),
        annual_precip_molar=("time", ann_total_P_molar),
        N_growing_days=("time", ann_total_GD),
        N_days=("time", n_days.to_numpy()),
        annual_mean_ca_in_GS=("time", annual_mean_ca_in_GS),
        annual_mean_chi_in_GS=("time", annual_mean_chi_in_GS),
        annual_mean_VPD_in_GS=("time", annual_mean_vpd_in_GS),
    ),
    coords=dict(time=n_days.time.dt.year.to_numpy()),
)

# Constants
z = 12.227  # leaf costs, mol m2 year
k = 0.5  # light extinction coefficient, -
f_0 = 0.65 * np.exp(-0.604169 * np.log(aridity_index.to_numpy() / 1.9) ** 2)
sigma = 0.771

# Convert mean GPP from µg C m-2 s-1 to annual moles
subdaily_annual_values["ann_total_A0_subdaily_smstress"] = (
    subdaily_annual_values["ann_mean_subdaily_gpp_smstress"]
    * (subdaily_annual_values["N_days"] * 24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)

subdaily_annual_values["ann_total_A0_subdaily"] = (
    subdaily_annual_values["ann_mean_subdaily_gpp"]
    * (subdaily_annual_values["N_days"] * 24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)

# Calculate fapar max using smstress'd A0

subdaily_annual_values["energy_limited_fapar"] = 1 - z / (
    k * subdaily_annual_values["ann_total_A0_subdaily_smstress"]
)
subdaily_annual_values["water_limited_fapar"] = (
    subdaily_annual_values["annual_mean_ca_in_GS"]
    * (1 - subdaily_annual_values["annual_mean_chi_in_GS"])
    / (1.6 * subdaily_annual_values["annual_mean_VPD_in_GS"])
) * (
    (f_0 * ann_total_P_molar) / subdaily_annual_values["ann_total_A0_subdaily_smstress"]
)

subdaily_annual_values["fapar_max"] = np.minimum(
    subdaily_annual_values["energy_limited_fapar"],
    subdaily_annual_values["water_limited_fapar"],
)
subdaily_annual_values["lai_max"] = -(1 / k) * np.log(
    1 - subdaily_annual_values["fapar_max"]
)

# Calculate ratio of steady state LAI to steady state GPP
subdaily_annual_values["m"] = (
    sigma * ann_total_GD * subdaily_annual_values["lai_max"]
) / (
    subdaily_annual_values["ann_total_A0_subdaily_smstress"]
    * subdaily_annual_values["fapar_max"]
)

# Now calculate the daily time series of LAI, which needs the daily assimilation
subdaily_daily_values = (
    subdaily_outputs[["PMod_gpp_smstress"]].resample(time="1D").mean()
)

# daily molar assimilation: mean gpp µC m-2 s-1 --> mol C m-2 day)
subdaily_daily_values["daily_A0"] = (
    subdaily_daily_values["PMod_gpp_smstress"]
    * (24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)


# Calculate steady state LAI, using principal branch of Lambert W function
#  - Map annual m and LAI values onto daily values of assimilation for each year
subdaily_daily_values = subdaily_daily_values.assign(
    annual_m=(
        "time",
        subdaily_annual_values["m"]
        .sel(time=subdaily_daily_values["time"].dt.year)
        .to_numpy(),
    ),
    annual_lai_max=(
        "time",
        subdaily_annual_values["lai_max"]
        .sel(time=subdaily_daily_values["time"].dt.year)
        .to_numpy(),
    ),
)

# Calculate daily mu value as m * daily molar assimilation:
mu = (subdaily_daily_values["annual_m"] * subdaily_daily_values["daily_A0"]).data

# Calculate the Lambert W0 value
Ls_term_1 = mu + (1 / k) * lambertw(-k * mu * np.exp(-k * mu), k=0)

# Check that all imaginary parts are zero or np.nan
if not np.all(np.logical_or(np.imag(Ls_term_1) == 0, np.isnan(Ls_term_1))):
    raise ValueError("Imaginary parts of Lambert W calculation are not zero")

# Clip the real parts at zero
Ls_term_1 = np.clip(np.real(Ls_term_1), a_min=0, a_max=None)

# Find the daily minimum of the lambert term and annual maximum LAI
Ls_daily = np.minimum(Ls_term_1, subdaily_daily_values["annual_lai_max"].data)

# Apply lagging
Ls_daily_lagged = acclim.apply_acclimation(Ls_daily)

# Save predicted daily time series for L
subdaily_daily_values["Ls_daily"] = Ls_daily
subdaily_daily_values = subdaily_daily_values.assign(
    Ls_daily_lagged=("time", Ls_daily_lagged)
)

# Save data to CSV - would use NetCDF for > 1 site.
# - Use float format to reduce file size and remove spurious precision.
# - Saving PModel inputs to make it easier to validate and rerun in regression testing

Path("subdaily_example").mkdir(exist_ok=True)

subdaily_outputs.to_pandas().to_csv(
    "subdaily_example/half_hourly_data.csv", float_format="%0.7g"
)

# Reduce the daily values to the core values to check.

subdaily_daily_values = subdaily_daily_values.drop_vars(
    [
        "PMod_chi",
        "PMod_ci",
        "ca",
        "tc",
        "co2",
        "ppfd",
        "vpd",
        "aet",
        "pre",
        "pet",
        "lat",
        "lon",
        "year",
        "annual_m",
    ]
)
subdaily_daily_values.to_pandas().to_csv(
    "subdaily_example/daily_outputs.csv", float_format="%0.7g"
)

subdaily_annual_values = subdaily_annual_values.drop_vars(["lat", "lon"])
subdaily_annual_values.to_pandas().to_csv(
    "subdaily_example/annual_outputs.csv", float_format="%0.7g"
)


# --------------------------------------------------------------------------------------
# Fortnightly model with no soil moisture stress
# --------------------------------------------------------------------------------------

# Resample half hourly data to fortnightly means
fortnight_resampler = de_gri_hh_xr.resample(time="2W")
fortnight_means = fortnight_resampler.mean()
fortnight_sum = fortnight_resampler.sum()
fortnight_resampler_from_daily = subdaily_daily_values.resample(time="2W")

# Extract the variables needed to run the model
fortnightly_outputs = xr.Dataset(
    data_vars={
        "tc_mean": fortnight_means["TA_F"],
        "vpd_mean": fortnight_means["VPD_F"],
        "patm_mean": fortnight_means["PA_F"],
        "co2_mean": fortnight_means["CO2_F_MDS"],
        "ppfd_mean": fortnight_means["PPFD"],
        "precip_molar_sum": fortnight_sum["P_F_MOLAR"],
        "year": fortnight_means.time.dt.year,
    }
)

env_fortnight = PModelEnvironment(
    tc=fortnightly_outputs["tc_mean"].to_numpy(),
    vpd=fortnightly_outputs["vpd_mean"].to_numpy(),
    patm=fortnightly_outputs["patm_mean"].to_numpy(),
    co2=fortnightly_outputs["co2_mean"].to_numpy(),
    ppfd=fortnightly_outputs["ppfd_mean"].to_numpy(),
    fapar=np.ones_like(fortnightly_outputs["tc_mean"]),
)

pmod_fortnight = PModel(env=env_fortnight)


# Save PModel variables and assign an arbitrary growing season vector
fortnightly_outputs = fortnightly_outputs.assign(
    dict(
        gpp=("time", pmod_fortnight.gpp),
        chi=("time", pmod_fortnight.optchi.chi),
        ci=("time", pmod_fortnight.optchi.ci),
        ca=("time", pmod_fortnight.env.ca),
        growing_season=("time", (fortnightly_outputs["tc_mean"] > 0.0).data),
    )
)

fortnightly_outputs = fortnightly_outputs.set_coords("year")

# TODO - allow endpoint to truncate.
avc2 = AnnualValueCalculator(
    timing=fortnightly_outputs["time"].to_numpy(),
    growing_season=fortnightly_outputs["growing_season"].to_numpy(),
    # endpoint=np.datetime64("2015-01-01"),
)


# Annual values - total precipitation and mean GPP
fortnightly_annual_precip = avc2.get_annual_totals(
    fortnightly_outputs["precip_molar_sum"].to_numpy()
)

fortnightly_annual_mean_gpp = avc2.get_annual_means(
    fortnightly_outputs["gpp"].to_numpy()
)

# Growing season means

fortnightly_annual_mean_vpd_gs = avc2.get_annual_means(
    fortnightly_outputs["vpd_mean"].to_numpy(), within_growing_season=True
)

fortnightly_annual_mean_chi_gs = avc2.get_annual_means(
    fortnightly_outputs["chi"].to_numpy(), within_growing_season=True
)

fortnightly_annual_mean_ca_gs = avc2.get_annual_means(
    fortnightly_outputs["ca"].to_numpy(), within_growing_season=True
)


# Create an annual dataset
fortnightly_annual_values = xr.Dataset(
    data_vars=dict(
        ann_mean_subdaily_gpp=("time", fortnightly_annual_mean_gpp),
        annual_precip_molar=("time", fortnightly_annual_precip),
        # N_growing_days=("time", ann_total_GD),
        # N_days=("time", n_days.to_numpy()),
        annual_mean_ca_in_GS=("time", fortnightly_annual_mean_ca_gs),
        annual_mean_chi_in_GS=("time", fortnightly_annual_mean_chi_gs),
        annual_mean_VPD_in_GS=("time", fortnightly_annual_mean_vpd_gs),
    ),
    # coords=dict(time=n_days.time.dt.year.to_numpy()),
)

# Convert mean GPP from µg C m-2 s-1 to annual moles
fortnightly_annual_values["ann_total_A0"] = (
    fortnightly_annual_values["gpp"]
    * (fortnightly_annual_values["N_days"] * 24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)


# fapar limits
fortnightly_annual_values["energy_limited_fapar"] = 1 - z / (
    k * fortnightly_annual_values["ann_total_A0"]
)
fortnightly_annual_values["water_limited_fapar"] = (
    (fortnightly_annual_values["ca"] * (1 - fortnightly_annual_values["chi"]))
    / (1.6 * fortnightly_annual_values["vpd_mean"])
    * (
        (f_0 * fortnightly_annual_values["precip_molar_sum"])
        / fortnightly_annual_values["ann_total_A0"]
    )
)

fortnightly_annual_values["fapar_max"] = np.minimum(
    fortnightly_annual_values["energy_limited_fapar"],
    fortnightly_annual_values["water_limited_fapar"],
)
fortnightly_annual_values["lai_max"] = -(1 / k) * np.log(
    1 - fortnightly_annual_values["fapar_max"]
)

# Save data to files

Path("fortnightly_example").mkdir(exist_ok=True)

fortnightly_outputs.to_pandas().to_csv(
    "fortnightly_example/fortnightly_data.csv", float_format="%0.7g"
)

fortnightly_annual_values.to_pandas().to_csv(
    "fortnightly_example/annual_outputs.csv", float_format="%0.7g"
)
