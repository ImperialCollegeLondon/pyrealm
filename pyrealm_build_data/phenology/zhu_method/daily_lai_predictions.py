"""This file is used to generate the Zhu daily predictions."""

import json

import numpy as np
import pandas as pd

from .plmodel_timeseries import plmodel_timeseries

daily_data = pd.read_csv("../fortnightly_example/daily_outputs.csv")
fapar_max = pd.read_csv("zhu_annual_fapar_max_from_fortnightly_data.csv")

with open("../DE-GRI_site_data.json") as site_file:
    site_data = json.load(site_file)

daily_data["time"] = pd.to_datetime(daily_data["time"])
daily_data["year"] = daily_data["time"].dt.year

daily_data = pd.merge(daily_data, fapar_max)

arr1, arr2, res_dict = plmodel_timeseries(
    A0_input=daily_data["daily_A0"].to_numpy()[None, None, :],
    fapar_max_input=daily_data["zhu_fapar_max"].to_numpy()[None, None, :],
    alpha_aet_pet=np.array([[site_data["AI"]]]),
)

results = pd.DataFrame(
    {
        "raw_LAI": res_dict["SLAI15DE1"].ravel(),
        "lagged_LAI": arr1.ravel(),
        "lagged_fAPAR": arr2.ravel(),
    }
)
