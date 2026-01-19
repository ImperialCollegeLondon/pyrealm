# mypy: ignore-errors
"""Zhu implementation of maximum fAPAR.

This file contains functionality taken from handover notes provided by Mateusz
Lisiewski, which included the `cal_fapar` function used by Ziqi Zhu to calculate maximum
fAPAR from the water and energy limited values. The contents are taken from the handover
notebooks.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from plmodel_timeseries import cal_fapar


def cal_fapar_actually_in_python(
    fapar_carbon: NDArray[np.floating], fapar_water: NDArray[np.floating], budyko=4
) -> NDArray[np.floating]:
    """Refactor of plmodel_timeseries.cal_fapar using array calculations."""
    cw_ratio = fapar_carbon / (np.clip(fapar_water, min=np.finfo(float).eps, max=None))
    return ((1 + cw_ratio) - (1 + cw_ratio**budyko) ** (1 / budyko)) * fapar_water


data = pd.read_csv("../fortnightly_example/annual_outputs.csv")
data = {k: v.to_numpy() for k, v in data.items()}

# Ziqi's values for f0 and zcost.
f0 = 0.5
zcost = 17


# Calculate energy-limited fapar and water-limited fapar separately (Cai et al., 2025)
fapar_carbon_calc = 1 - (zcost / (0.5 * data["ann_total_A0"]))
fapar_water_calc = (
    (
        (
            data["annual_mean_ca_in_GS"]
            / (1.6 * data["annual_mean_VPD_in_GS"] * data["ann_total_A0"])
        )
        * (1 - data["annual_mean_chi_in_GS"])
    )
    * f0
    * data["annual_precip_molar"]
)

fapar_max = cal_fapar(fapar_carbon=fapar_carbon_calc, fapar_water=fapar_water_calc)

fapar_max2 = cal_fapar_actually_in_python(
    fapar_carbon=fapar_carbon_calc, fapar_water=fapar_water_calc
)

assert np.allclose(fapar_max, fapar_max2)

outputs = pd.DataFrame(
    {
        "year": data["time"].astype("datetime64[Y]").astype("str").astype("int"),
        "zhu_fapar_max": fapar_max,
    }
)

outputs.to_csv("zhu_annual_fapar_max_from_fortnightly_data.csv")
