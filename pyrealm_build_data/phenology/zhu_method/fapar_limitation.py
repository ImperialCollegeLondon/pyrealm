# mypy: ignore-errors
"""Zhu implementation of maximum fAPAR.

This file contains functionality taken from handover notes provided by Mateusz
Lisiewski, which included the `cal_fapar` function used by Ziqi Zhu to calculate maximum
fAPAR from the water and energy limited values. The contents are taken from the handover
notebooks.
"""

import numpy as np
import pandas as pd


# ziqi's function
def cal_fapar(fapar_carbon, fapar_water, options=None):
    """Function to calculate maximum fapar from water and energy limited versions."""
    # Handle options
    if options is None:
        options = {}

    const_budyko = options.get("const_budyko", 4)

    # Convert to float arrays
    fapar_carbon = np.asarray(fapar_carbon, dtype=float)
    fapar_water = np.asarray(fapar_water, dtype=float)

    # Handle scalar broadcasting
    if fapar_carbon.ndim == 0 and fapar_water.ndim > 0:
        fapar_carbon = np.full_like(fapar_water, fapar_carbon)
    elif fapar_water.ndim == 0 and fapar_carbon.ndim > 0:
        fapar_water = np.full_like(fapar_carbon, fapar_water)
    elif fapar_carbon.shape != fapar_water.shape:
        try:
            # Test if broadcasting works
            _ = fapar_carbon + fapar_water
        except ValueError:
            raise ValueError(
                "fapar_carbon and fapar_water must be scalar or equal-sized arrays"
            )

    # Store original shape
    sza = fapar_carbon.shape

    # Flatten arrays
    flatC = fapar_carbon.ravel()
    flatW = fapar_water.ravel()
    n = flatC.size

    # Initialize output arrays
    flat_out = np.full(n, np.nan)
    flat_ratio = np.full(n, np.nan)
    flat_factor = np.full(n, np.nan)

    safety_eps = np.finfo(float).eps

    # Loop through each element
    for ii in range(n):
        fc = flatC[ii]
        fw = flatW[ii]

        # Safe denominator
        denom_safe = fw + (fw == 0) * safety_eps
        r_local = fc / denom_safe
        flat_ratio[ii] = r_local

        one_plus_r = 1 + r_local
        r_clamped = max(r_local, -0.999)
        r_pow = r_clamped**const_budyko
        inside = 1 + r_pow
        root_term = inside ** (1 / const_budyko)
        fcomb = one_plus_r - root_term

        flat_factor[ii] = fcomb
        flat_out[ii] = fcomb * fw

    # Reshape to original shape
    fapar_max = flat_out.reshape(sza)

    return fapar_max


def cal_fapar_actually_in_python(fapar_carbon, fapar_water, budyko=4):
    """Refactor using array calculations."""
    cw_ratio = fapar_carbon / (np.clip(fapar_water, min=np.finfo(float).eps, max=None))
    return ((1 + cw_ratio) - (1 + cw_ratio**budyko) ** (1 / budyko)) * fapar_water


data = pd.read_csv("../fortnightly_example/annual_outputs.csv")

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

outputs = pd.DataFrame({"year": data["time"], "zhu_fapar_max": fapar_max})

outputs.to_csv("zhu_annual_fapar_max_from_fortnightly_data.csv")
