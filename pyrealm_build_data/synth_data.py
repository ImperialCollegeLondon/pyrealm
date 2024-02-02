"""This script uses a parametrized model to compress the input dataset.

It fits a time series model to the input data and stores the model parameters,
which can be later used to reconstruct the original dataset.
"""
import numpy as np
import pandas as pd
import xarray as xr


def make_time_features(t: np.ndarray, const_only: bool = False) -> pd.DataFrame:
    """Make time features for a given time index."""
    dt = pd.to_datetime(t).rename("time")
    df = sm.add_constant(pd.DataFrame(index=dt))

    if const_only:
        return df

    # Linear trend
    df["linear"] = (dt - pd.Timestamp("2000-01-01")) / pd.Timedelta("1h")

    # El-Nino and La-Nina seasonality component
    for k in [2, 3, 4, 6]:
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


def fit_ts_model(da: xr.DataArray, const_only: bool = False) -> pd.DataFrame:
    """Fit a time series model to the data."""
    df = da.resample(time="2h").to_series().unstack("time").T
    df = df.dropna(axis=1, how="all")
    features = make_time_features(df.index, const_only=const_only)
    params = pd.DataFrame(index=df.columns, columns=features.columns)
    for k in tqdm(df.columns):
        y = df[k].dropna()
        X = features.loc[y.index]
        model = sm.OLS(y, X).fit()
        params.loc[k] = model.params
    return params.astype(float)


def decompress(ds: xr.Dataset) -> xr.Dataset:
    """Decompress the dataset from the model parameters."""
    return xr.Dataset({k: a @ ds["feature"] for k, a in ds.items() if k != "feature"})


if __name__ == "__main__":
    import statsmodels.api as sm
    from tqdm import tqdm

    ds = xr.open_dataset("inputs_data_24.25.nc")

    const_vars_over_time = ["patm"]
    const_vars_over_space = ["co2"]

    params_ds = xr.Dataset()

    for k in ds.data_vars:
        print("Fitting", k)
        if k in const_vars_over_space:
            da = ds[k].isel(lat=[0], lon=[0])
        else:
            da = ds[k]
        if k in const_vars_over_time:
            ps = fit_ts_model(da, const_only=True)
        else:
            ps = fit_ts_model(da)
        params_ds[k] = ps.to_xarray().to_dataarray()

    for k, a in params_ds.items():
        if k in const_vars_over_space:
            params_ds[k] = a.fillna(a.isel(lat=0, lon=0))
        elif k in const_vars_over_time:
            params_ds[k] = a.fillna(0)

    params_ds["feature"] = make_time_features(ds.time).to_xarray().to_dataarray()
    params_ds.to_netcdf("data_model_params.nc")
