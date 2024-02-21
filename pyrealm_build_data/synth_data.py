"""This script uses a parametrized model to compress the input dataset.

It fits a time series model to the input data and stores the model parameters.
The dataset can then be reconstructed from the model parameters using the `reconstruct`
function, provided with a custom time index.
"""
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike


def make_time_features(t: ArrayLike) -> pd.DataFrame:
    """Make time features for a given time index.

    The model can be written as
    g(t) = a₀ + a₁ t + ∑_i b_i sin(2π f_i t) + c_i cos(2π f_i t),
    where t is the time index, f_i are the frequencies, and a₀, a₁, b_i, c_i are the
    model parameters.

    Args:
        t: An array of datetime values.
    """
    dt = pd.to_datetime(t).rename("time")
    df = pd.DataFrame(index=dt).assign(const=1.0)

    df["linear"] = (dt - pd.Timestamp("2000-01-01")) / pd.Timedelta("365.25d")

    for f in [730.5, 365.25, 12, 6, 4, 3, 2, 1, 1 / 2, 1 / 3, 1 / 4, 1 / 6]:
        df[f"freq_{f:.2f}_sin"] = np.sin(2 * np.pi * f * df["linear"])
        df[f"freq_{f:.2f}_cos"] = np.cos(2 * np.pi * f * df["linear"])

    return df


def fit_ts_model(da: xr.DataArray, fs: pd.DataFrame) -> xr.DataArray:
    """Fit a time series model to the data.

    Args:
        da: A DataArray with the input data.
        fs: A DataFrame with the time features.
    """
    print("Fitting", da.name)

    da = da.isel(time=slice(None, None, 4))  # downsample along time
    da = da.dropna("time", how="all")
    da = da.fillna(da.mean("time"))
    df = da.to_series().unstack("time").T

    Y = df.values  # (times, locs)
    X = fs.loc[df.index].values  # (times, feats)
    A, res, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (feats, locs)

    loss = np.mean(res) / len(X) / np.var(Y)
    pars = pd.DataFrame(A.T, index=df.columns, columns=fs.columns)

    print("Loss:", loss)
    return pars.to_xarray().to_dataarray()


def reconstruct(
    ds: xr.Dataset, dt: ArrayLike, bounds: dict | None = None
) -> xr.Dataset:
    """Reconstruct the full dataset from the model parameters.

    Args:
        ds: A Dataset with the model parameters.
        dt: An array of datetime values.
        bounds: A dictionary with the bounds for the reconstructed variables.
    """
    if bounds is None:
        bounds = dict(
            temp=(-25, 80),
            patm=(3e4, 11e4),
            vpd=(0, 1e4),
            co2=(0, 1e3),
            fapar=(0, 1),
            ppfd=(0, 1e4),
        )
    x = make_time_features(dt).to_xarray().to_dataarray()
    ds = xr.Dataset({k: a @ x for k, a in ds.items()})
    ds = xr.Dataset({k: a.clip(*bounds[k]) for k, a in ds.items()})
    return ds


if __name__ == "__main__":
    ds = xr.open_dataset("pyrealm_build_data/inputs_data_24.25.nc")

    # drop locations with all NaNs (for any variable)
    mask = ~ds.isnull().all("time").to_dataarray().any("variable")
    ds = ds.where(mask, drop=True)

    special_time_features = dict(
        patm=["const"],
        co2=["const", "linear"],
    )

    features = make_time_features(ds.time)

    model = xr.Dataset()
    for k in ds.data_vars:
        cols = special_time_features.get(k, features.columns)
        model[k] = fit_ts_model(ds[k], features[cols])

    model = model.fillna(0.0)
    model.to_netcdf("pyrealm_build_data/data_model.nc")
