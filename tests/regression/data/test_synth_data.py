"""Test the quality of the synthetic data generated from the model parameters."""

import numpy as np
import pytest
import xarray as xr


def r2_score(y_true: xr.DataArray, y_pred: xr.DataArray) -> float:
    """Compute the R2 score."""
    SSE = ((y_true - y_pred) ** 2).sum().item()
    SST = ((y_true - y_true.mean()) ** 2).sum().item()
    return 1 - SSE / SST


@pytest.fixture
def dataset(path="pyrealm_build_data/inputs_data_24.25.nc"):
    """The original dataset."""
    return xr.open_dataset(path)


@pytest.fixture
def synth_dataset(path="pyrealm_build_data/data_model_params.nc"):
    """Generate the synthetic dataset from the model parameters."""
    ds = xr.open_dataset(path)
    return xr.Dataset({k: a @ ds["feature"] for k, a in ds.items() if k != "feature"})


def test_synth_data_quality(dataset, synth_dataset):
    """Test the quality of the synthetic data."""
    sample = np.random.choice(dataset.time.size, 1000, replace=False)
    for k in dataset.data_vars:
        t = dataset[k].sel(lat=synth_dataset.lat).isel(time=sample)
        p = synth_dataset[k].isel(time=sample)
        r2 = r2_score(t, p)
        print(f"R2 score for {k}: {r2:.2f}")
        assert r2 > 0.85
