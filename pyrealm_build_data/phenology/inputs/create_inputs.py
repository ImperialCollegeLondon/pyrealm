"""Calculate fortnightly and subdaily inputs for phenology regression tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from run_length_encode import run_length_encode  # type: ignore[import-not-found]
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

from pyrealm.core.time_series import AnnualValueCalculator
from pyrealm.core.water import convert_water_mm_to_moles
from pyrealm.pmodel import (
    AcclimationModel,
    PModel,
    PModelEnvironment,
    SubdailyPModel,
)
from pyrealm.pmodel.functions import calculate_soilmstress_mengoli

# Set the format used for writing floats to file
FLOAT_FORMAT = "%0.8g"

# --------------------------------------------------------------------------------------
# Site data - calculate aridity as PET / P over 20 years.
# --------------------------------------------------------------------------------------

# Load the site data
with open("source/DE-GRI_site_data.json") as dpath:
    de_gri_site_data = json.load(dpath)

# Load Soil moisture from SPLASH run on CRU TS 4.07
de_gri_splash = xr.load_dataset("source/DE_gri_splash_cru_ts4.07_2000_2019.nc")

# Calculate 20 year aridity index (2000 - 2020) as PET/P
aridity_index = de_gri_splash["pet"].mean() / de_gri_splash["pre"].mean()
# And aet_pet_ratio
aet_pet_ratio = de_gri_splash["aet"].mean() / de_gri_splash["pet"].mean()

# Store those in the site data
with open("source/DE-GRI_site_data.json", "w") as dpath:
    de_gri_site_data["AI_from_cruts"] = float(aridity_index)
    de_gri_site_data["aet_pet_ratio"] = float(aet_pet_ratio)
    json.dump(obj=de_gri_site_data, fp=dpath)

# --------------------------------------------------------------------------------------
# Data preparation - this section loads the raw FluxNet data and prepares it for use at
# the half hourly scale:
#
# 1. Extract forcing data for PModel from Fluxnet
# 2. Add molar precipitation data from FluxNet
# 3. Add gpp penalties from soil moisture stress, from CRU Splash data
# 4. Add growing season boolean data.
# --------------------------------------------------------------------------------------

# Load the half hourly data - ignoring mypy's dislike of perfectly functional numeric
# inputs to na_values.
de_gri_source = pd.read_csv(
    "source/DE_GRI_hh_fluxnet_simple.csv",
    na_values=["-9999-9999.0", -9999.0, -9999],  # type: ignore[list-item]
)

# Calculate time as np.datetime64, set as the index
de_gri_source["time"] = pd.to_datetime(
    de_gri_source["TIMESTAMP_START"], format="%Y%m%d%H%M"
)
de_gri_source = de_gri_source.set_index("time")

# --------------------------------------------------------------------------------------
# SUBDAILY PMODEL WITH SOIL MOISTURE STRESS
# --------------------------------------------------------------------------------------

pmodel_inputs_hh = pd.DataFrame(
    dict(
        tc=de_gri_source["TA_F"].clip(lower=-25),  #  Clip temperatures at -25°C
        vpd=de_gri_source["VPD_F"] * 100,  # VPD from hPa to Pa
        patm=de_gri_source["PA_F"] * 1000,  # Pressure from kPa to Pa
        co2=de_gri_source["CO2_F_MDS"],
        ppfd=de_gri_source["SW_IN_F_MDS"] * 2.04,  # PPFD from SWDOWN
        fapar=1,  # Models estimating potential GPP
    )
)

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
pmodel_inputs_hh["precip_molar"] = convert_water_mm_to_moles(
    water_mm=de_gri_source["P_F"].to_numpy(),
    tc=de_gri_source["TA_F"].to_numpy(),
    patm=de_gri_source["PA_F"].to_numpy(),
)

# We also want to support posthoc GPP penalties, so calculate the Mengoli daily stress
# factors using SPLASH soil moisture / bucket size
soilm_stress = pd.DataFrame(
    dict(
        date=pd.to_datetime(de_gri_splash["time"].to_numpy()),
        soilm_stress=calculate_soilmstress_mengoli(
            soilm=de_gri_splash["wn"].to_numpy() / 150,
            aridity_index=aridity_index.to_numpy(),
        ),
    )
)

# We need a definition of growing season, defined by identifying which days in the time
# series have a mean temperature of greater than freezing, and where that continued for
# at least 5 days.

# Get the raw non-freezing days
growing_season = pd.DataFrame(
    dict(growing_day_raw=pmodel_inputs_hh["tc"].resample("1D").mean() > 0)
)

# Eliminate short chunks of above-zero temperatures.
gsl_lengths, gsl_values = run_length_encode(
    growing_season["growing_day_raw"].to_numpy()
)
gsl_values[np.logical_and(gsl_values == 1, gsl_lengths < 5)] = 0
growing_season["growing_season"] = np.repeat(gsl_values, gsl_lengths)

# Merge those two daily variables on the date, which duplicates values to the half
# hourly time scale and also reduces the SPLASH 20 year daily values run to the DE_GRI
# hh rows. This mucks around with the indexing, so reset to keep time and then reinstate

pmodel_inputs_hh = pmodel_inputs_hh.reset_index()
pmodel_inputs_hh["date"] = pd.to_datetime(pmodel_inputs_hh["time"].dt.date)
pmodel_inputs_hh = pmodel_inputs_hh.merge(soilm_stress)
pmodel_inputs_hh = pmodel_inputs_hh.merge(
    growing_season.drop(columns="growing_day_raw"), left_on="date", right_on="time"
)
pmodel_inputs_hh = pmodel_inputs_hh.set_index("time")


# Export PModel inputs - truncating float precision using "%g" to drop trailing zeros.
(Path(".") / "subdaily").mkdir(exist_ok=True)
pmodel_inputs_hh.drop(columns="date").to_csv(
    "subdaily/pmodel_inputs.csv", float_format=FLOAT_FORMAT
)

# --------------------------------------------------------------------------------------
# Get PModel predictions from subdaily inputs
# --------------------------------------------------------------------------------------

# Calculate the PModel photosynthetic environment

env = PModelEnvironment(
    tc=pmodel_inputs_hh["tc"].to_numpy(),
    vpd=pmodel_inputs_hh["vpd"].to_numpy(),
    co2=pmodel_inputs_hh["co2"].to_numpy(),
    patm=pmodel_inputs_hh["patm"].to_numpy(),
    fapar=pmodel_inputs_hh["fapar"].to_numpy(),
    ppfd=pmodel_inputs_hh["ppfd"].to_numpy(),
)

# Set up the datetimes of the observations and set the acclimation window
acclim = AcclimationModel(
    datetimes=pmodel_inputs_hh.index.to_numpy(), alpha=1 / 15, allow_holdover=True
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

# Get a dataframe of PModel predictions at half hourly scale
pmodel_outputs_hh = pd.DataFrame(
    dict(
        gpp=subdaily_pmodel.gpp,
        gpp_smstress=subdaily_pmodel.gpp * pmodel_inputs_hh["soilm_stress"],
        chi=subdaily_pmodel.optchi.chi,
        ci=subdaily_pmodel.optchi.ci,
        ca=subdaily_pmodel.env.ca,
    ),
    index=pmodel_inputs_hh.index,
)

pmodel_outputs_hh.to_csv("subdaily/pmodel_outputs.csv", float_format=FLOAT_FORMAT)

# --------------------------------------------------------------------------------------
# Calculate annual values for testing FaparLimitation
# --------------------------------------------------------------------------------------

# Calculate annual values
avc_hh = AnnualValueCalculator(
    data_shape=env.tc.shape,
    timing=acclim,
    subset_mask=pmodel_inputs_hh["growing_season"].to_numpy(),
)


# Average annual GPP ± soil moisture stress
# GPP, precipitation and growing day totals across the whole year.

annual_values_hh = pd.DataFrame(
    dict(
        year=pd.Series(avc_hh.years).dt.year,
        N_growing_days=avc_hh.year_n_days_subset,
        N_days=avc_hh.year_n_days,
        annual_mean_gpp=avc_hh.get_annual_means(pmodel_outputs_hh["gpp"].to_numpy()),
        annual_mean_gpp_smstress=avc_hh.get_annual_means(
            pmodel_outputs_hh["gpp_smstress"].to_numpy()
        ),
        annual_precip_molar=avc_hh.get_annual_totals(
            pmodel_inputs_hh["precip_molar"].to_numpy()
        ),
        # Chi, ca and VPD in growing season
        annual_mean_ca_in_GS=avc_hh.get_annual_means(
            pmodel_outputs_hh["ca"].to_numpy(), within_subset=True
        ),
        annual_mean_chi_in_GS=avc_hh.get_annual_means(
            pmodel_outputs_hh["chi"].to_numpy(), within_subset=True
        ),
        annual_mean_VPD_in_GS=avc_hh.get_annual_means(
            pmodel_inputs_hh["vpd"].to_numpy(), within_subset=True
        ),
    ),
)


# Convert mean GPP from µg C m-2 s-1 to annual moles
annual_values_hh["annual_total_A0_smstress"] = (
    annual_values_hh["annual_mean_gpp_smstress"]
    * (annual_values_hh["N_days"] * 24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)

annual_values_hh["annual_total_A0"] = (
    annual_values_hh["annual_mean_gpp"]
    * (annual_values_hh["N_days"] * 24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)

annual_values_hh.to_csv("subdaily/annual_inputs.csv", float_format=FLOAT_FORMAT)

# --------------------------------------------------------------------------------------
# Calculate daily assimilation values for testing Phenology
# --------------------------------------------------------------------------------------

# Now calculate the daily assimilation for calculating LAI time series
# daily molar assimilation: mean gpp µC m-2 s-1 --> mol C m-2 day)
daily_gpp = pmodel_outputs_hh["gpp_smstress"].resample("1D").mean()
daily_productivity_hh = pd.DataFrame(
    dict(
        daily_gpp=daily_gpp,
        daily_A0=daily_gpp * (24 * 60 * 60 * 1e-6) / env.core_const.k_c_molmass,
    )
)

daily_productivity_hh.to_csv(
    "subdaily/daily_assimilation.csv", float_format=FLOAT_FORMAT
)

# --------------------------------------------------------------------------------------
# FORTNIGHTLY MODEL WITH NO SOIL MOISTURE STRESS
# --------------------------------------------------------------------------------------

# Resample half hourly data to fortnightly means
fortnight_resampler = pmodel_inputs_hh.drop(columns="date").resample("14D")
fortnight_means = fortnight_resampler.mean()
fortnight_sum = fortnight_resampler.sum()

# Extract the variables needed to run the model
pmodel_inputs_ft = pd.DataFrame(
    dict(
        tc=fortnight_means["tc"],
        vpd=fortnight_means["vpd"],
        patm=fortnight_means["patm"],
        co2=fortnight_means["co2"],
        ppfd=fortnight_means["ppfd"],
        fapar=1,
        precip_molar=fortnight_sum["precip_molar"],
    )
)

pmodel_inputs_ft["growing_season"] = pmodel_inputs_ft["tc"] > 0

# Export PModel inputs - truncating float precision using "%g" to drop trailing zeros.
(Path(".") / "fortnightly").mkdir(exist_ok=True)
pmodel_inputs_ft.to_csv("fortnightly/pmodel_inputs.csv", float_format=FLOAT_FORMAT)

# --------------------------------------------------------------------------------------
# Get PModel predictions from fortnightly inputs
# --------------------------------------------------------------------------------------

env_fortnight = PModelEnvironment(
    tc=pmodel_inputs_ft["tc"].to_numpy(),
    vpd=pmodel_inputs_ft["vpd"].to_numpy(),
    patm=pmodel_inputs_ft["patm"].to_numpy(),
    co2=pmodel_inputs_ft["co2"].to_numpy(),
    ppfd=pmodel_inputs_ft["ppfd"].to_numpy(),
    fapar=pmodel_inputs_ft["fapar"].to_numpy(),
)

fortnight_pmodel = PModel(env=env_fortnight)

pmodel_outputs_ft = pd.DataFrame(
    dict(
        gpp=fortnight_pmodel.gpp,
        chi=fortnight_pmodel.optchi.chi,
        ci=fortnight_pmodel.optchi.ci,
        ca=fortnight_pmodel.env.ca,
    ),
    index=pmodel_inputs_ft.index,
)

pmodel_outputs_ft.to_csv("fortnightly/pmodel_outputs.csv", float_format=FLOAT_FORMAT)

# --------------------------------------------------------------------------------------
# Calculate annual values for testing FaparLimitation
# --------------------------------------------------------------------------------------

avc_ft = AnnualValueCalculator(
    data_shape=fortnight_pmodel.env.shape,
    timing=pmodel_outputs_ft.index.to_numpy(),
    subset_mask=pmodel_inputs_ft["growing_season"].to_numpy(),
)

# GPP, precipitation and growing day totals across the whole year.
annual_values_ft = pd.DataFrame(
    dict(
        year=pd.Series(avc_ft.years).dt.year,
        N_growing_days=avc_ft.year_n_days_subset,
        N_days=avc_ft.year_n_days,
        annual_mean_gpp=avc_ft.get_annual_means(pmodel_outputs_ft["gpp"].to_numpy()),
        annual_precip_molar=avc_ft.get_annual_totals(
            pmodel_inputs_ft["precip_molar"].to_numpy()
        ),
        # Chi, ca and VPD in growing season
        annual_mean_ca_in_GS=avc_ft.get_annual_means(
            pmodel_outputs_ft["ca"].to_numpy(), within_subset=True
        ),
        annual_mean_chi_in_GS=avc_ft.get_annual_means(
            pmodel_outputs_ft["chi"].to_numpy(), within_subset=True
        ),
        annual_mean_VPD_in_GS=avc_ft.get_annual_means(
            pmodel_inputs_ft["vpd"].to_numpy(), within_subset=True
        ),
    ),
)


# Convert mean GPP from µg C m-2 s-1 to annual moles
annual_values_ft["annual_total_A0"] = (
    annual_values_ft["annual_mean_gpp"]
    * (annual_values_ft["N_days"] * 24 * 60 * 60 * 1e-6)
    / env.core_const.k_c_molmass
)

annual_values_ft.to_csv(
    "fortnightly/annual_inputs.csv", float_format=FLOAT_FORMAT, index=False
)

# --------------------------------------------------------------------------------------
# Calculate daily assimilation values for testing Phenology
# --------------------------------------------------------------------------------------

# Get a daily phenology time series by interpolation

gpp = pmodel_outputs_ft["gpp"].to_numpy()
time = pmodel_outputs_ft.index.to_numpy()
time_int = time.astype(np.int_)


# The interp1d object cannot be called with datetime64 values as new_x
interpolator = interp1d(time_int, gpp)
daily_timestamps = np.arange(
    time[0], time[-1] + np.timedelta64(1, "D"), np.timedelta64(1, "D")
)
daily_timestamps_int = daily_timestamps.astype(np.int_)
daily_gpp = interpolator(daily_timestamps_int)


daily_productivity_ft = pd.DataFrame(
    dict(
        time=daily_timestamps,
        daily_gpp=daily_gpp,
        daily_A0=daily_gpp * (24 * 60 * 60 * 1e-6) / env.core_const.k_c_molmass,
    )
)

daily_productivity_ft.to_csv(
    "fortnightly/daily_assimilation.csv", float_format=FLOAT_FORMAT, index=False
)
