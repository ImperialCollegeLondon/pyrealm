"""This script uses a parametrized model to compress the input dataset.

It fits a time series model to the input data and stores the model parameters,
which can be later used to reconstruct the original dataset.
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


def fit_ts_model(df: pd.DataFrame, fs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Fit a time series model to the data."""
    from tqdm import tqdm
    from scipy.optimize import minimize, LinearConstraint

    df = df.dropna(axis=1, how="all")  # drop locations with all NaNs
    df = df.interpolate(method="time")  # fill NaNs with linear interpolation
    T, M, D = df.shape[0], df.shape[1], fs.shape[1]  # times, locs, feats
    X = fs.values  # (T, D)
    Y = df.values  # (T, M)
    cons = LinearConstraint(X, Y.min(axis=0), Y.max(axis=0))
    res = minimize(lambda a: np.sum((X @ a.reshape(D, M) - Y)**2), np.zeros(M * D), constraints=cons)
    if res.success:
        pars = pd.DataFrame(res.x.reshape(D, M), index=df.columns, columns=fs.columns)
        losses = np.mean((X @ pars.values - Y) ** 2, axis=0) / np.var(Y, axis=0)
        return pars, pd.Series(losses, index=df.columns)


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
        ps, ls = fit_ts_model(df, fs)  # (locations, features)
        print("Loss:", ls.mean())
        ps[features.keys().difference(ps.columns)] = 0.0
        model[k] = ps.to_xarray().to_dataarray()

    # model["features"] = features.to_xarray().to_dataarray()
    model.to_netcdf("pyrealm_build_data/data_model.nc")
