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
    datetimes=de_gri_hh_xr["time"].to_numpy(),
    alpha=1 / 15,
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

# Get an xarray dataset of the required outputs at half hourly scale to be resampled to
# the daily scale and also the input variables required to fit the model
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

# Resample those pmodel outputs to daily frequency and get mean values
subdaily_daily_resampler = subdaily_outputs.resample(time="1D")
subdaily_daily_values = subdaily_daily_resampler.mean()

# Add splash data to the daily data, using left join to subset 20 year time series to
# only the observed site days.
subdaily_daily_values = subdaily_daily_values.merge(de_gri_splash, join="left")

# Calculate soil moisture stress factor using SPLASH soil moisture / bucket size
soilm_stress = calc_soilmstress_mengoli(
    soilm=subdaily_daily_values["wn"].to_numpy() / 150,
    aridity_index=aridity_index.to_numpy(),
)

subdaily_daily_values = subdaily_daily_values.assign(
    soilm_stress=("time", soilm_stress)
)

# Apply the soil moisture penalty to the daily mean GPP
subdaily_daily_values["PMod_gpp_smstress"] = (
    subdaily_daily_values["PMod_gpp"] * subdaily_daily_values["soilm_stress"]
)

# Calculate growing seasons - note that this is operating over the _whole_ timespan, so
# includes blocks that cross year boundaries

# Get the growing days
subdaily_daily_values["growing_day"] = subdaily_daily_values["tc"] > 0

# Eliminate short chunks of growth
gsl_lengths, gsl_values = run_length_encode(
    subdaily_daily_values["growing_day"].to_numpy()
)
gsl_values[np.logical_and(gsl_values == 1, gsl_lengths < 5)] = 0
gsl_filtered = np.repeat(gsl_values, gsl_lengths)

# Save filtered growing days
subdaily_daily_values = subdaily_daily_values.assign(
    growing_day_filtered=("time", gsl_filtered)
)

# Calculate annual values

# Precipitation and number of growing days

ann_total_P_molar = de_gri_hh_xr["P_F_MOLAR"].groupby("time.year").sum()
ann_total_GD = subdaily_daily_values["growing_day_filtered"].groupby("time.year").sum()
ann_days_in_year = (
    subdaily_daily_values["growing_day_filtered"].groupby("time.year").count()
)

# Average annual GPP ± soil moisture stress
# GPP, precipitation and growing day totals across the whole year.
ann_mean_subdaily_gpp = subdaily_daily_values["PMod_gpp"].groupby("time.year").mean()
ann_mean_subdaily_gpp_smstress = (
    subdaily_daily_values["PMod_gpp_smstress"].groupby("time.year").mean()
)

# Average conditions within growing days
growing_conditions = (
    subdaily_daily_values[["ca", "PMod_chi", "vpd", "time"]]
    .where(subdaily_daily_values["growing_day_filtered"], drop=True)
    .groupby("time.year")
    .mean()
    .rename(
        {
            "ca": "annual_mean_ca_in_GS",
            "PMod_chi": "annual_mean_chi_in_GS",
            "vpd": "annual_mean_VPD_in_GS",
        }
    )
)

# Create an annual dataset, joining on site data to drop extra CRU years
subdaily_annual_values = xr.merge(
    [
        ann_mean_subdaily_gpp_smstress.rename("ann_mean_subdaily_gpp_smstress"),
        ann_mean_subdaily_gpp.rename("ann_mean_subdaily_gpp"),
        ann_total_P_molar.rename("annual_precip_molar"),
        ann_total_GD.rename("N_growing_days"),
        ann_days_in_year.rename("N_days"),
        growing_conditions,
    ],
    join="left",
)


# Constants
z = 12.227  # leaf costs, mol m2 year
k = 0.5  # light extinction coefficient, -
f_0 = 0.65 * np.exp(-0.604169 * np.log(aridity_index / 1.9) ** 2)
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


# Calculate steady state LAI, using principal branch of Lambert W function
#  - Map annual m and LAI values onto each year
subdaily_daily_values["annual_m"] = subdaily_annual_values["m"].sel(
    year=subdaily_daily_values["time"].dt.year
)
subdaily_daily_values["annual_lai_max"] = subdaily_annual_values["lai_max"].sel(
    year=subdaily_daily_values["time"].dt.year
)

# Calculate daily mu value as m * daily molar assimilation:
#   mean gpp µC m-2 s-1 --> mol Cm-2 day)
mu = (
    subdaily_daily_values["annual_m"]
    * subdaily_daily_values["PMod_gpp_smstress"]
    * (24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)

# Calculate the Lambert W0 value, screen for non-zero imaginary parts, clip at zero
Ls_term_1 = mu + (1 / k) * lambertw(-k * mu * np.exp(-k * mu), k=0)

if not np.all(np.imag(Ls_term_1.data) == 0):
    raise ValueError("Imaginary parts of Lambert W calculation are not zero")

Ls_term_1 = np.clip(np.real(Ls_term_1), a_min=0, a_max=None)

# Find the daily minimum of the lambert term and annual maximum LAI
# Ls_daily = xr.ufuncs.minimum(Ls_term_1, de_gri_daily_values["annual_lai_max"])
Ls_daily = np.minimum(Ls_term_1, subdaily_daily_values["annual_lai_max"])

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
fortnight_resampler = subdaily_outputs.resample(time="2W")
fortnight_means = fortnight_resampler.mean()
fortnight_sum = fortnight_resampler.sum()

# Extract the variables needed to run the model
fortnightly_outputs = xr.Dataset(
    data_vars={
        "tc_mean": fortnight_means["tc"],
        "vpd_mean": fortnight_means["vpd"],
        "patm_mean": fortnight_means["patm"],
        "co2_mean": fortnight_means["co2"],
        "ppfd_mean": fortnight_means["ppfd"],
        "precip_molar_sum": fortnight_sum["precip_molar"],
        "year": fortnight_means.time.dt.year,
    }
)

# Drop 2015 week
fortnightly_outputs = fortnightly_outputs.where(
    fortnightly_outputs.year < 2015, drop=True
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

# Annual values - total precipitation and mean GPP
fortnightly_annual_precip = (
    fortnightly_outputs[["precip_molar_sum", "year"]].groupby("year").sum()
)

fortnightly_annual_mean_gpp = (
    fortnightly_outputs[["gpp", "year"]].groupby("year").mean()
)

# Growing season means
fortnightly_growing_season = fortnightly_outputs.where(
    fortnightly_outputs.growing_season, drop=True
)

fortnightly_annual_gs_means = (
    fortnightly_growing_season[["vpd_mean", "ca", "chi", "year"]].groupby("year").mean()
)

fortnightly_annual_values = xr.merge(
    [
        fortnightly_annual_precip,
        fortnightly_annual_mean_gpp,
        fortnightly_annual_gs_means,
        subdaily_annual_values["N_days"],
    ]
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
