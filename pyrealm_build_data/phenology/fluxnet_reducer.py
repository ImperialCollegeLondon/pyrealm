"""Minimize regression test inputs.

This script reduces the full FluxNET file for the DE-GRI site to only the variables
needed for regression testing (300 MB -> 10MB). The full dataset is not stored in
pyrealm_build_data.
"""

import pandas as pd

# Load the full FluxNET file and set NAs - mypy objects to numeric values but they are
# in fact handled by pandas
de_gri_flux_hh = pd.read_csv(
    "FLX_DE-Gri_FLUXNET2015_FULLSET_HH_2004-2014_1-4.csv",
    na_values=["-9999", "-9999.0", -9999, -9999.0],  # type: ignore[list-item]
)

# Reduce to the P Model forcing variables plus precipitation
de_gri_flux_hh_subset = de_gri_flux_hh[
    [
        "TIMESTAMP_START",
        "TA_F",
        "VPD_F",
        "PA_F",
        "CO2_F_MDS",
        "SW_IN_F_MDS",
        "P_F",
    ]
]

# Save to CSV
de_gri_flux_hh_subset.to_csv("DE_GRI_hh_fluxnet_simple.csv")
