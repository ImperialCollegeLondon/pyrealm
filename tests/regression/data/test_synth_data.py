"""Test the quality of the synthetic data generated from the model parameters."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    DATASET = xr.open_dataset("pyrealm_build_data/inputs_data_24.25.nc")
except ValueError:
    pytest.skip("Original LFS dataset not checked out.")

VARS = DATASET.data_vars


def r2_score(y_true: xr.DataArray, y_pred: xr.DataArray) -> float:
    """Compute the R2 score."""
    SSE = ((y_true - y_pred) ** 2).sum()
    SST = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - SSE / SST


@pytest.fixture
def syndata(modelpath="pyrealm_build_data/data_model.nc"):
    """The synthetic dataset."""
    from pyrealm_build_data.synth_data import reconstruct

    model = xr.open_dataset(modelpath)
    ts = pd.date_range("2012-01-01", "2018-01-01", freq="12h")
    return reconstruct(model, ts)


@pytest.fixture
def dataset(syndata):
    """The original dataset."""
    return DATASET.sel(time=syndata.time)


@pytest.mark.parametrize("var", VARS)
def test_synth_data_quality(dataset, syndata, var):
    """Test the quality of the synthetic data."""
    times = syndata.time[np.random.choice(syndata.time.size, 1000, replace=False)]
    lats = syndata.lat[np.random.choice(syndata.lat.size, 100, replace=False)]
    t = dataset[var].sel(lat=lats, time=times)
    p = syndata[var].sel(lat=lats, time=times)
    s = r2_score(t, p)
    print(f"R2 score for {var} is {s:.2f}")
    if s < 0.8:
        # import matplotlib.pyplot as plt
        # t.sortby("time").mean(["lat", "lon"]).plot()
        # p.sortby("time").mean(["lat", "lon"]).plot()
        # plt.show()
        raise AssertionError
