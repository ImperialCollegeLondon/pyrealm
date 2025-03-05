"""Methods for input and output of test data for FaparLimitation class."""

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from xarray import DataArray


def write_faparlim_input(
    ann_total_A0_subdaily_penalised: xr.Dataset,
    ann_total_P_fnet: xr.Dataset,
    aridity_index: DataArray,
    annual_values: xr.Dataset,
) -> None:
    """Method for writing out the data needed for the FaparLimitation constructor."""

    with Dataset("faparlim_input.nc", "w") as f:
        f.createDimension("year", len(ann_total_A0_subdaily_penalised.year.data))
        year = f.createVariable("year", "i", ("year",))
        year[:] = ann_total_A0_subdaily_penalised.year.data

        annual_total_A0_subdaily = f.createVariable(
            "annual_total_A0_subdaily", float, ("year",)
        )
        annual_total_A0_subdaily[:] = ann_total_A0_subdaily_penalised.data

        annual_total_P = f.createVariable("annual_total_P", float, ("year",))
        annual_total_P[:] = ann_total_P_fnet.data

        f.createVariable("aridity_idx", float, ())
        f.variables["aridity_idx"][...] = aridity_index.data

    annual_values.to_netcdf("faparlim_input.nc", mode="a")


def write_pmodel_faparlim_input(
    tc: DataArray,
    vpd: DataArray,
    co2: DataArray,
    patm: DataArray,
    growing_season: DataArray,
    datetimes: DataArray,
    precipitation: DataArray,
    aridity_index: DataArray,
    ppfd: DataArray,
) -> None:
    """Writing out the data needed for the from_pmodel FaparLimitation class method."""

    with Dataset("faparlim_pmodel_input.nc", "w") as f:
        f.createDimension("datatimes_length", 29)

        f.createDimension("datetimes", len(datetimes))
        datetimes_var = f.createVariable("datetimes", str, ("datetimes",))
        datetimes_var[:] = datetimes[:].astype(str)

        f.createDimension("days", len(growing_season))
        growing_season_var = f.createVariable("growing_season", "i", ("days",))
        growing_season_var[:] = growing_season

        tc_var = f.createVariable("tc", float, ("datetimes",))
        tc_var[:] = tc

        vpd_var = f.createVariable("vpd", float, ("datetimes",))
        vpd_var[:] = vpd

        co2_var = f.createVariable("co2", float, ("datetimes",))
        co2_var[:] = co2

        patm_var = f.createVariable("patm", float, ("datetimes",))
        patm_var[:] = patm

        precip_var = f.createVariable("precipitation", float, ("days",))
        precip_var[:] = precipitation

        f.createVariable("aridity_idx", float, ())
        f.variables["aridity_idx"][...] = aridity_index.data

        ppfd_var = f.createVariable("ppfd", float, ("datetimes",))
        ppfd_var[:] = ppfd


def read_faparlim_input(
    filename: str,
) -> tuple[xr.Dataset, xr.Dataset, DataArray, DataArray, DataArray, DataArray]:
    """Method for reading in the data needed for the FaparLimitation constructor."""

    with Dataset(filename, "r") as f:
        annual_total_A0_subdaily = f.variables["annual_total_A0_subdaily"][...].data
        annual_total_P = f.variables["annual_total_P"][...].data
        aridity_index = f.variables["aridity_idx"][...].data
        annual_mean_ca = f.variables["annual_mean_ca_in_GS"][...].data
        annual_mean_chi = f.variables["annual_mean_chi_in_GS"][...].data
        annual_mean_vpd = f.variables["annual_mean_VPD_in_GS"][...].data

    return (
        annual_total_A0_subdaily,
        annual_total_P,
        aridity_index,
        annual_mean_ca,
        annual_mean_chi,
        annual_mean_vpd,
    )


def read_pmodel_faparlim_input(
    filename: str,
) -> tuple[
    DataArray,
    DataArray,
    DataArray,
    DataArray,
    DataArray,
    DataArray,
    DataArray,
    DataArray,
    DataArray,
]:
    """Reading in the data needed for the from_pmodel FaparLimitation class method."""

    with Dataset(filename, "r") as f:
        tc = f.variables["tc"][...].data
        vpd = f.variables["vpd"][...].data
        co2 = f.variables["co2"][...].data
        patm = f.variables["patm"][...].data
        growing_season = f.variables["growing_season"][...].data
        datetimes = f.variables["datetimes"][...].astype(np.datetime64)
        precipitation = f.variables["precipitation"][...].data
        aridity_index = f.variables["aridity_idx"][...].data
        ppfd = f.variables["ppfd"][...].data

    return (
        tc,
        vpd,
        co2,
        patm,
        growing_season,
        datetimes,
        precipitation,
        aridity_index,
        ppfd,
    )
