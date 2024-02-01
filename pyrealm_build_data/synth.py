"""This script uses a parametrized model to compress the input dataset.

It fits a time series model to the input data and stores the model parameters,
which can be later used to reconstruct the original dataset.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from np.typing import ArrayLike
from tqdm import tqdm

ds = xr.open_dataset("inputs_data_24.25.nc")

const_vars_over_time = ["patm"]
const_vars_over_space = ["co2"]


def make_time_features(dt: ArrayLike, is_const: bool = False) -> pd.DataFrame:
    """Make time features for a given time index."""
    dt = pd.to_datetime(dt).rename("time")
    df = sm.add_constant(pd.DataFrame(index=dt))

    if is_const:
        return df

    # Linear trend
    df["linear"] = (dt - pd.Timestamp("2000-01-01")) / pd.Timedelta("1h")

    # El-Nino and La-Nina seasonality component
    for k in [1, 2, 3, 4]:
        df[f"cross_year_{k}_sin"] = np.sin(2 * k * np.pi * dt.year / 12)
        df[f"cross_year_{k}_cos"] = np.cos(2 * k * np.pi * dt.year / 12)

    # Yearly seasonality component
    for k in [1, 2, 3, 4]:
        df[f"yearly_{k}_sin"] = np.sin(2 * k * np.pi * dt.dayofyear / 365)
        df[f"yearly_{k}_cos"] = np.cos(2 * k * np.pi * dt.dayofyear / 365)

    # Daily seasonality component
    for k in [1, 2, 3, 4]:
        df[f"daily_{k}_sin"] = np.sin(2 * k * np.pi * dt.hour / 24)
        df[f"daily_{k}_cos"] = np.cos(2 * k * np.pi * dt.hour / 24)

    return df


def fit_ts_model(da: xr.DataArray, is_const: bool = False) -> pd.DataFrame:
    """Fit a time series model to the data."""
    df = da.sel(time=slice(None, None, 2)).to_series().unstack("time").T
    df = df.dropna(axis=1, how="all")
    features = make_time_features(df.index, is_const=is_const)
    params = pd.DataFrame(index=df.columns, columns=features.columns)
    for k in tqdm(df.columns):
        y = df[k].dropna()
        X = features.loc[y.index]
        model = sm.OLS(y, X).fit()
        params.loc[k] = model.params
    return params.astype(float)


params_ds = xr.Dataset()

for k in ds.data_vars:
    print("Fitting", k)
    if k in const_vars_over_space:
        da = ds[k].isel(lat=[0], lon=[0])
    else:
        da = ds[k]
    if k in const_vars_over_time:
        ps = fit_ts_model(da, is_const=True)
    else:
        ps = fit_ts_model(da)
    params_ds[k] = ps.to_xarray().to_dataarray()

for k, a in params_ds.items():
    if k in const_vars_over_space:
        params_ds[k] = a.fillna(a.isel(lat=0, lon=0))
    elif k in const_vars_over_time:
        params_ds[k] = a.fillna(0)

fs = make_time_features(ds.time).to_xarray().to_dataarray()
params_ds["feature"] = fs
params_ds.to_netcdf("data_model_params.nc")


def reconstruct_data(params_ds: xr.Dataset) -> xr.Dataset:
    """Reconstruct the data from the model parameters."""
    rec_ds = xr.Dataset()
    f = params_ds["feature"]
    for k, a in params_ds.items():
        if k != "feature":
            rec_ds[k] = xr.dot(f, a)
    return rec_ds


rec_ds = reconstruct_data(params_ds)


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the R2 score."""
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


for k in ds.data_vars:
    t = ds[k].sel(lat=rec_ds.lat).squeeze()[::321]
    p = rec_ds[k].squeeze()[::321]
    r2 = r2_score(t, p)
    print(f"R2 score for {k}: {r2:.2f}")
