"""This script uses a parametrized model to compress the input dataset.

It fits a time series model to the input data and stores the model parameters.
The dataset can then be reconstructed from the model parameters using the `reconstruct` function,
provided with a custom time index.
"""
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr


def make_time_features(t: np.ndarray) -> pd.DataFrame:
    """Make time features for a given time index."""
    dt = pd.to_datetime(t).rename("time")
    df = pd.DataFrame(index=dt).assign(const=1.0)

    df["linear"] = (dt - pd.Timestamp("2000-01-01")) / pd.Timedelta("365.25d")

    for f in [730.5, 365.25, 12, 6, 4, 3, 2, 1, 1 / 2, 1 / 3, 1 / 4, 1 / 6]:
        df[f"freq_{f:.2f}_sin"] = np.sin(2 * np.pi * f * df["linear"])
        df[f"freq_{f:.2f}_cos"] = np.cos(2 * np.pi * f * df["linear"])

    return df


def fit_ts_model(df: pd.DataFrame, fs: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Fit a time series model to the data."""
    df = df.dropna(axis=1, how="all").fillna(df.mean())
    Y = df.values  # (times, locs)
    X = fs.values  # (times, feats)
    A = np.linalg.pinv(X) @ Y  # (feats, locs)
    loss = np.mean((X @ A - Y) ** 2) / np.var(Y)
    pars = pd.DataFrame(A.T, index=df.columns, columns=fs.columns)
    return pars, loss


def reconstruct(ds: xr.Dataset, dt: np.ndarray | pd.DatetimeIndex) -> xr.Dataset:
    """Reconstruct the full dataset from the model parameters."""
    x = make_time_features(dt).to_xarray().to_dataarray()
    return xr.Dataset({k: a @ x for k, a in ds.items() if k != "features"})


if __name__ == "__main__":
    ds = xr.open_dataset("pyrealm_build_data/inputs_data_24.25.nc")

    mask = ~ds.isnull().all("time").to_dataarray().any("variable")
    ds = ds.where(mask, drop=True)

    special_time_features = dict(
        patm=["const"],
        co2=["const", "linear"],
    )

    features = make_time_features(ds.time)
    model = xr.Dataset()

    for k in ds.data_vars:
        print("Fitting", k)
        da = ds[k].isel(time=slice(None, None, 4))  # downsample along time
        df = da.to_series().unstack("time").T  # (datetimes, locations)
        fs = features.loc[df.index]  # (datetimes, features)
        fs = fs[special_time_features.get(k, fs.columns)]
        ps, r = fit_ts_model(df, fs)  # (locations, features)
        print("Loss:", r)
        ps[features.keys().difference(ps.columns)] = 0.0
        model[k] = ps.to_xarray().to_dataarray()

    model.to_netcdf("pyrealm_build_data/data_model.nc")
