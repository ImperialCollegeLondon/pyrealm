# mypy: ignore-errors
"""Zhu implementation of phenology.

This file contains functionality taken from handover notes provided by Mateusz
Lisiewski, which included the `cal_fapar` function used by Ziqi Zhu to calculate maximum
fAPAR from the water and energy limited values. It also provided the
`plmodel_timeseries` function used to create phenology time series predictions. These
are included as the original version in the `plmodel_timeseries` module.
"""

import json

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from plmodel_timeseries import cal_fapar, plmodel_timeseries


def cal_fapar_actually_in_python(
    fapar_carbon: NDArray[np.floating], fapar_water: NDArray[np.floating], budyko=4
) -> NDArray[np.floating]:
    """Refactor of plmodel_timeseries.cal_fapar using array calculations."""
    cw_ratio = fapar_carbon / (np.clip(fapar_water, min=np.finfo(float).eps, max=None))
    return ((1 + cw_ratio) - (1 + cw_ratio**budyko) ** (1 / budyko)) * fapar_water


# Ziqi's values for k, f0 and zcost.
k = 0.5
f0 = 0.5
zcost = 17

# Load site data
with open("../inputs/source/DE-GRI_site_data.json") as site_file:
    site_data = json.load(site_file)


# --------------------------------------------------------------------------------------
# Calculate fapar max and LAI for annual values from fortnightly inputs
# --------------------------------------------------------------------------------------

data = pd.read_csv("../inputs/fortnightly/annual_inputs.csv")
data = {k: v.to_numpy() for k, v in data.items()}

# Calculate energy-limited fapar and water-limited fapar separately (Cai et al., 2025)
fapar_carbon_calc = 1 - (zcost / (0.5 * data["annual_total_A0"]))
fapar_water_calc = (
    (
        (
            data["annual_mean_ca_in_GS"]
            / (1.6 * data["annual_mean_VPD_in_GS"] * data["annual_total_A0"])
        )
        * (1 - data["annual_mean_chi_in_GS"])
    )
    * f0
    * data["annual_precip_molar"]
)

fapar_max_ft = cal_fapar(fapar_carbon=fapar_carbon_calc, fapar_water=fapar_water_calc)

fapar_max_ft2 = cal_fapar_actually_in_python(
    fapar_carbon=fapar_carbon_calc, fapar_water=fapar_water_calc
)

assert np.allclose(fapar_max_ft, fapar_max_ft2)


# --------------------------------------------------------------------------------------
# Calculate fapar max and LAI from annual values from subdaily inputs
# --------------------------------------------------------------------------------------

data = pd.read_csv("../inputs/subdaily/annual_inputs.csv")
data = {k: v.to_numpy() for k, v in data.items()}

# Calculate energy-limited fapar and water-limited fapar separately (Cai et al., 2025)
fapar_carbon_calc = 1 - (zcost / (0.5 * data["annual_total_A0_smstress"]))
fapar_water_calc = (
    (
        (
            data["annual_mean_ca_in_GS"]
            / (1.6 * data["annual_mean_VPD_in_GS"] * data["annual_total_A0_smstress"])
        )
        * (1 - data["annual_mean_chi_in_GS"])
    )
    * f0
    * data["annual_precip_molar"]
)

fapar_max_hh = cal_fapar(fapar_carbon=fapar_carbon_calc, fapar_water=fapar_water_calc)

fapar_max_hh2 = cal_fapar_actually_in_python(
    fapar_carbon=fapar_carbon_calc, fapar_water=fapar_water_calc
)

assert np.allclose(fapar_max_hh, fapar_max_hh2)


outputs = pd.DataFrame(
    dict(
        year=data["year"],
        fapar_max_ft=fapar_max_ft,
        lai_max_ft=-(1 / k) * np.log(1 - fapar_max_ft),
        fapar_max_hh=fapar_max_hh,
        lai_max_hh=-(1 / k) * np.log(1 - fapar_max_hh),
    )
)

outputs.to_csv("fapar_max_predictions.csv", index=False)

# --------------------------------------------------------------------------------------
# Calculate daily LAI  from fortnightly inputs
# --------------------------------------------------------------------------------------

daily_data_ft = pd.read_csv("../inputs/fortnightly/daily_assimilation.csv")

daily_data_ft["time"] = pd.to_datetime(daily_data_ft["time"])
daily_data_ft["year"] = daily_data_ft["time"].dt.year

# Merge in annual fapar max
daily_data_ft = pd.merge(daily_data_ft, outputs[["year", "fapar_max_ft"]])


arr1_ft, _, res_dict_ft = plmodel_timeseries(
    A0_input=daily_data_ft["daily_A0"].to_numpy()[None, None, :],
    fapar_max_input=daily_data_ft["fapar_max_ft"].to_numpy()[None, None, :],
    alpha_aet_pet=np.array([[site_data["aet_pet_ratio"]]]),
)


# --------------------------------------------------------------------------------------
# Calculate daily LAI  from subdaily inputs
# --------------------------------------------------------------------------------------

daily_data_hh = pd.read_csv("../inputs/subdaily/daily_assimilation.csv")

daily_data_hh["time"] = pd.to_datetime(daily_data_hh["time"])
daily_data_hh["year"] = daily_data_hh["time"].dt.year

# Merge in annual fapar max
daily_data_hh = pd.merge(daily_data_hh, outputs[["year", "fapar_max_hh"]])


arr1_hh, _, res_dict_hh = plmodel_timeseries(
    A0_input=daily_data_hh["daily_A0"].to_numpy()[None, None, :],
    fapar_max_input=daily_data_hh["fapar_max_hh"].to_numpy()[None, None, :],
    alpha_aet_pet=np.array([[site_data["aet_pet_ratio"]]]),
)

# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------

# Save predicted daily time series for L
lai_predictions_ft = pd.DataFrame(
    dict(
        time=daily_data_ft["time"],
        Ls_daily_ft=res_dict_ft["SLAI15DE1"].ravel(),
        Ls_daily_lagged_ft=arr1_ft.ravel(),
    )
)

lai_predictions_hh = pd.DataFrame(
    dict(
        time=daily_data_hh["time"],
        Ls_daily_hh=res_dict_hh["SLAI15DE1"].ravel(),
        Ls_daily_lagged_hh=arr1_hh.ravel(),
    )
)

lai_predictions = lai_predictions_hh.merge(lai_predictions_ft, how="left")
lai_predictions.to_csv("daily_lai_predictions.csv", float_format="%0.8g", index=False)
