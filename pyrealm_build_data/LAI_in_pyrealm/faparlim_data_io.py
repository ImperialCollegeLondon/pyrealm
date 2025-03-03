"""Methods for input and output of test data for FaparLimitation class."""

import xarray as xr
from scipy.io import netcdf_file
from xarray import DataArray


def write_faparlim_input(
    ann_total_A0_subdaily_penalised: xr.Dataset,
    ann_total_P_fnet: xr.Dataset,
    aridity_index: DataArray,
    annual_values: xr.Dataset,
) -> None:
    """Method for writing out the data needed for the FaparLimitation constructor."""

    with netcdf_file("faparlim_input.nc", "w") as f:
        f.createDimension("year", len(ann_total_A0_subdaily_penalised.year.data))
        year = f.createVariable("year", "i", ("year",))
        year[:] = ann_total_A0_subdaily_penalised.year.data

        annual_total_A0_subdaily = f.createVariable(
            "annual_total_A0_subdaily", type=float, dimensions=("year",)
        )
        annual_total_A0_subdaily[:] = ann_total_A0_subdaily_penalised.data

        annual_total_P = f.createVariable(
            "annual_total_P", type=float, dimensions=("year",)
        )
        annual_total_P[:] = ann_total_P_fnet.data

        aridity_idx = f.createVariable("aridity_idx", type=float, dimensions=())
        aridity_idx[:] = aridity_index.data

        f.close()

    annual_values.to_netcdf("faparlim_input.nc", mode="a")


def read_faparlim_input(
    filename: str,
) -> tuple[xr.Dataset, xr.Dataset, DataArray, DataArray, DataArray, DataArray]:
    """Method for reading in the data needed for the FaparLimitation constructor."""

    with netcdf_file(filename, "r") as f:
        annual_total_A0_subdaily = f.variables["annual_total_A0_subdaily"].data
        annual_total_P = f.variables["annual_total_P"].data
        aridity_index = f.variables["aridity_idx"].data
        annual_mean_ca = f.variables["annual_mean_ca_in_GS"].data
        annual_mean_chi = f.variables["annual_mean_chi_in_GS"].data
        annual_mean_vpd = f.variables["annual_mean_VPD_in_GS"].data
        f.close()

    return (
        annual_total_A0_subdaily,
        annual_total_P,
        aridity_index,
        annual_mean_ca,
        annual_mean_chi,
        annual_mean_vpd,
    )
