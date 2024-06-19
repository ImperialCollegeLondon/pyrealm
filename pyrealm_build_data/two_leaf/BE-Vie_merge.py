"""Simple script to merge BE-Vie FLUXNet site data files."""

from pathlib import Path

import pandas as pd

# Load and prepare 4 datasets to merge on their time information

# 1) FLUXNet data for most environmental data and measured GPP
#    - Half hourly sampling
#    - Some missing night-time / dawn data in PPFD
fluxnet_data = pd.read_csv(Path("FLX_BE-Vie_2014.csv"), na_values="-9999")
fluxnet_data["time"] = pd.to_datetime(
    fluxnet_data["TIMESTAMP_START"], format="%Y%m%d%H%M"
)

# 2) Leaf area index data
#    - Half hourly sampling
lai_data = pd.read_csv(Path("data_LAI_BE-Vie_2014.csv"))
lai_data["time"] = pd.to_datetime(fluxnet_data["TIME"])

# 3) FAPAR dataset:
#    - Daily data
#    - Resample to half hourly frequency and forward fill for a fixed daily value.
#    - Need to add the next day to resample through to the end of the time series.
fapar_data = pd.read_csv(Path("data_Beni2_BE-Vie_2014.csv"))
fapar_data = pd.concat([fapar_data, pd.DataFrame.from_records([{"TIME": 20150101}])])
fapar_data["day"] = pd.to_datetime(fapar_data["TIME"], format="%Y%m%d")
fapar_data = fapar_data.set_index(fapar_data["day"])
fapar_halfhour = fapar_data.resample("30 min").ffill()
fapar_halfhour["time"] = fapar_halfhour.index

# 4) Giulia GPP predictions
#    - Half hourly
#    - Resample to half hourly frequency and forward fill for a fixed daily value.
#    - Need to add the next day to resample through to the end of the time series.
giulia_subdaily_ggp = pd.read_csv(Path("../subdaily/subdaily_BE_Vie_2014.csv"))
giulia_subdaily_ggp["time"] = pd.to_datetime(fluxnet_data["time"])

# Reduce to key variables and merge
fluxnet_data = fluxnet_data[
    ["time", "PPFD_IN", "VPD_F", "TA_F", "CO2_F_MDS", "PA_F", "GPP_DT_CUT_REF"]
]
lai_data = lai_data[["time", "LAI"]]
fapar_halfhour = fapar_halfhour[["time", "fapar_spl"]]
giulia_subdaily_ggp = giulia_subdaily_ggp[["time", "GPP_JAMES"]]

forcing_data = pd.merge(fluxnet_data, lai_data, on="time")
forcing_data = pd.merge(forcing_data, fapar_halfhour, on="time")
forcing_data = pd.merge(forcing_data, giulia_subdaily_ggp, on="time")

# Rename variables
forcing_data = forcing_data.rename(
    columns={
        "PPFD_IN": "ppfd",
        "VPD_F": "vpd",
        "TA_F": "tc",
        "CO2_F_MDS": "co2",
        "PA_F": "patm",
        "GPP_DT_CUT_REF": "gpp_fluxnet",
        "fapar_spl": "fapar",
    }
)

# Fix units and convert to float values
forcing_data["vpd"] *= 100  # hPa to Pa
forcing_data["patm"] *= 1000  # kPa to Pa
forcing_data["ppfd"] = forcing_data["ppfd"].astype(float)

# Fill missing PPFD with zeros
forcing_data["ppfd"] = forcing_data["ppfd"].where(forcing_data["ppfd"].notna(), 0)

# Save merged file
forcing_data.to_csv("merged_BE-Vie_data.csv", index=False)
