"""Calculate annual fapar max and daily LAI.

This uses the Cai et al method to calculate annual maximum fapar and then Boya Zhou's
implementation of the daily phenology.
"""

import json

import numpy as np
import pandas as pd
from scipy.special import lambertw  # type: ignore[import-untyped]

from pyrealm.core.utilities import exponential_moving_average

# Load the site data
with open("../inputs/source/DE-GRI_site_data.json") as site_file:
    site_data = json.load(site_file)

# --------------------------------------------------------------------------------------
# MAXIMUM FAPAR PREDICTIONS USING SHIRLEY CAI'S METHOD AND PARAMETERISATION
# --------------------------------------------------------------------------------------

# Constants
z = 12.227  # leaf costs, mol m2 year
k = 0.5  # light extinction coefficient, -
f_0 = 0.65 * np.exp(-0.604169 * np.log(site_data["AI"] / 1.9) ** 2)
sigma = 0.771

# Subdaily predictions - Calculate fapar max using smstress'd A0

annual_data_hh = pd.read_csv("../inputs/subdaily/annual_inputs.csv")

energy_limited_fapar_hh = 1 - z / (k * annual_data_hh["ann_total_A0_smstress"])
water_limited_fapar_hh = (
    annual_data_hh["annual_mean_ca_in_GS"]
    * (1 - annual_data_hh["annual_mean_chi_in_GS"])
    / (1.6 * annual_data_hh["annual_mean_VPD_in_GS"])
) * (
    (f_0 * annual_data_hh["annual_precip_molar"])
    / annual_data_hh["ann_total_A0_smstress"]
)

fapar_max_hh = np.minimum(
    energy_limited_fapar_hh,
    water_limited_fapar_hh,
)
lai_max_hh = -(1 / k) * np.log(1 - fapar_max_hh)

# Calculate ratio of steady state LAI to steady state GPP
steady_state_m_hh = (sigma * annual_data_hh["N_growing_days"] * lai_max_hh) / (
    annual_data_hh["ann_total_A0_smstress"] * fapar_max_hh
)

# Fortnightly predictions - calculate fapar max using fortnightly values

annual_data_ft = pd.read_csv("../inputs/fortnightly/annual_inputs.csv")


energy_limited_fapar_ft = 1 - z / (k * annual_data_ft["ann_total_A0"])
water_limited_fapar_ft = (
    annual_data_ft["annual_mean_ca_in_GS"]
    * (1 - annual_data_ft["annual_mean_chi_in_GS"])
    / (1.6 * annual_data_ft["annual_mean_VPD_in_GS"])
) * ((f_0 * annual_data_ft["annual_precip_molar"]) / annual_data_ft["ann_total_A0"])

fapar_max_ft = np.minimum(
    energy_limited_fapar_ft,
    water_limited_fapar_ft,
)
lai_max_ft = -(1 / k) * np.log(1 - fapar_max_ft)

# Calculate ratio of steady state LAI to steady state GPP
steady_state_m_ft = (sigma * annual_data_ft["N_growing_days"] * lai_max_ft) / (
    annual_data_ft["ann_total_A0"] * fapar_max_ft
)

## Write to file
cai_fapar_max = pd.DataFrame(
    dict(
        year=annual_data_ft.year,
        energy_limited_fapar_hh=energy_limited_fapar_hh,
        water_limited_fapar_hh=water_limited_fapar_hh,
        fapar_max_hh=fapar_max_hh,
        lai_max_hh=lai_max_hh,
        steady_state_m_hh=steady_state_m_hh,
        energy_limited_fapar_ft=energy_limited_fapar_ft,
        water_limited_fapar_ft=water_limited_fapar_ft,
        fapar_max_ft=fapar_max_ft,
        lai_max_ft=lai_max_ft,
        steady_state_m_ft=steady_state_m_ft,
    )
)

cai_fapar_max.to_csv("fapar_max_predictions.csv", float_format="%0.8g")

# --------------------------------------------------------------------------------------
# CALCULATE PHENOLOGY TIME SERIES USING BOYA ZHOU'S METHOD
# --------------------------------------------------------------------------------------

# Subdaily half hourly inputs

daily_assimilation_hh = pd.read_csv("../inputs/subdaily/daily_assimilation.csv")

#  - Map annual m and LAI values onto daily values of assimilation for each year
daily_assimilation_hh["time"] = pd.to_datetime(daily_assimilation_hh["time"])
daily_assimilation_hh["year"] = daily_assimilation_hh["time"].dt.year

daily_assimilation_hh = daily_assimilation_hh.merge(
    cai_fapar_max[["year", "lai_max_hh", "steady_state_m_hh"]]
)

# Calculate daily mu value as m * daily molar assimilation:
mu = daily_assimilation_hh["steady_state_m_hh"] * daily_assimilation_hh["daily_A0"]

# Calculate the Lambert W0 value
Ls_term_1 = mu + (1 / k) * lambertw(-k * mu * np.exp(-k * mu), k=0)

# Check that all imaginary parts are zero or np.nan
if not np.all(np.logical_or(np.imag(Ls_term_1) == 0, np.isnan(Ls_term_1))):
    raise ValueError("Imaginary parts of Lambert W calculation are not zero")

# Clip the real parts at zero
Ls_term_1 = np.clip(np.real(Ls_term_1), a_min=0, a_max=None)

# Find the daily minimum of the lambert term and annual maximum LAI, rounding to remove
# tiny values
Ls_daily_hh = np.minimum(Ls_term_1, daily_assimilation_hh["lai_max_hh"]).round(8)

# Apply exponential lag
Ls_daily_lagged_hh = exponential_moving_average(
    Ls_daily_hh, alpha=1 / 15, allow_holdover=True
)


# Fortnightly inputs

daily_assimilation_ft = pd.read_csv("../inputs/fortnightly/daily_assimilation.csv")

#  - Map annual m and LAI values onto daily values of assimilation for each year
daily_assimilation_ft["time"] = pd.to_datetime(daily_assimilation_ft["time"])
daily_assimilation_ft["year"] = daily_assimilation_ft["time"].dt.year

daily_assimilation_ft = daily_assimilation_ft.merge(
    cai_fapar_max[["year", "lai_max_ft", "steady_state_m_ft"]]
)

# Calculate daily mu value as m * daily molar assimilation:
mu = daily_assimilation_ft["steady_state_m_ft"] * daily_assimilation_ft["daily_A0"]

# Calculate the Lambert W0 value
Ls_term_1 = mu + (1 / k) * lambertw(-k * mu * np.exp(-k * mu), k=0)

# Check that all imaginary parts are zero or np.nan
if not np.all(np.logical_or(np.imag(Ls_term_1) == 0, np.isnan(Ls_term_1))):
    raise ValueError("Imaginary parts of Lambert W calculation are not zero")

# Clip the real parts at zero
Ls_term_1 = np.clip(np.real(Ls_term_1), a_min=0, a_max=None)

# Find the daily minimum of the lambert term and annual maximum LAI, rounding to remove
# tiny values
Ls_daily_ft = np.minimum(Ls_term_1, daily_assimilation_ft["lai_max_ft"]).round(8)


# Apply lagging
Ls_daily_lagged_ft = exponential_moving_average(
    Ls_daily_ft, alpha=1 / 15, allow_holdover=True
)

# Save predicted daily time series for L
lai_predictions_hh = pd.DataFrame(
    dict(
        time=daily_assimilation_hh["time"],
        Ls_daily_hh=Ls_daily_hh,
        Ls_daily_lagged_hh=Ls_daily_lagged_hh,
    )
)

lai_predictions_ft = pd.DataFrame(
    dict(
        time=daily_assimilation_ft["time"],
        Ls_daily_ft=Ls_daily_ft,
        Ls_daily_lagged_ft=Ls_daily_lagged_ft,
    )
)

lai_predictions = lai_predictions_hh.merge(lai_predictions_ft, how="left")
lai_predictions.to_csv("daily_lai_predictions.csv", float_format="%0.8g")
