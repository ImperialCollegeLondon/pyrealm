import numpy as np
import pandas as pd


def accumulate_daily(
    flux_subdaily:  np.ndarray,
    datetimes:      np.ndarray,
    timestep_sec:   float,
    flux_unit_in:   str = "umol_C_m2_s",
    flux_unit_out:  str = "gC_m2_day",
) -> pd.DataFrame:
    """
    Accumulate a subdaily flux to daily totals by integration.

    Multiplies each subdaily rate by the timestep duration (seconds)
    and sums within each calendar day:

        daily_total = sum(flux_i * timestep_sec)  for all i in day

    Unit conversions supported:
        "umol_C_m2_s"  → "umol_C_m2_day"  : multiply by timestep_sec
        "umol_C_m2_s"  → "gC_m2_day"      : multiply by timestep_sec * 12.011e-6
        "ugC_m2_s"     → "gC_m2_day"      : multiply by timestep_sec * 1e-6

    Parameters
    ----------
    flux_subdaily : np.ndarray, shape (n_timesteps,)
        Subdaily flux rate (e.g. GPP, Rd, or net carbon gain)
    datetimes     : np.ndarray of datetime64, shape (n_timesteps,)
        Subdaily datetime array corresponding to flux_subdaily
    timestep_sec  : float
        Duration of each subdaily timestep in seconds
        e.g. 1800.0 for 30-min, 3600.0 for hourly
    flux_unit_in  : str
        Unit of input flux:
        - "umol_C_m2_s"  : µmol C m-2 s-1  (GPP after unit conversion,
                            or Rd, or net carbon gain)
        - "ugC_m2_s"     : µg C m-2 s-1    (raw pyrealm GPP output)
    flux_unit_out : str
        Desired unit of daily output:
        - "umol_C_m2_day" : µmol C m-2 day-1
        - "gC_m2_day"     : g C m-2 day-1  (most common for carbon budgets)

    Returns
    -------
    df_daily : pd.DataFrame with columns:
        'date'        : calendar date
        'daily_total' : accumulated daily flux in flux_unit_out
        'n_timesteps' : number of valid (non-NaN) subdaily timesteps per day
                        (useful for QC — days with missing data will have < expected)

    Raises
    ------
    ValueError : if unsupported unit combination is requested

    Examples
    --------
    # GPP from pyrealm (µg C m-2 s-1) → g C m-2 day-1
    daily_GPP = accumulate_daily(
        flux_subdaily = subdaily_model.gpp,
        datetimes     = datetimes,
        timestep_sec  = 1800.0,
        flux_unit_in  = "ugC_m2_s",
        flux_unit_out = "gC_m2_day",
    )

    # Rd or net carbon gain (µmol C m-2 s-1) → g C m-2 day-1
    daily_Rd = accumulate_daily(
        flux_subdaily = Rd_subdaily,
        datetimes     = datetimes,
        timestep_sec  = 1800.0,
        flux_unit_in  = "umol_C_m2_s",
        flux_unit_out = "gC_m2_day",
    )
    """

    # ── Unit conversion factor ────────────────────────────────────────────────
    # Converts rate (per second) → total per timestep → daily total
    # in the desired output unit
    supported = {
        ("umol_C_m2_s", "umol_C_m2_day") : timestep_sec,
        ("umol_C_m2_s", "gC_m2_day")     : timestep_sec * 12.011e-6,
        ("ugC_m2_s",    "gC_m2_day")     : timestep_sec * 1e-6,
    }
    key = (flux_unit_in, flux_unit_out)
    if key not in supported:
        raise ValueError(
            f"Unsupported unit conversion: {flux_unit_in} → {flux_unit_out}.\n"
            f"Supported combinations: {list(supported.keys())}"
        )
    conversion_factor = supported[key]

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "date"  : pd.to_datetime(datetimes).normalize(),  # floor to calendar day
        "flux"  : flux_subdaily * conversion_factor,      # per-timestep amount
    })

    # ── Group by day and sum ──────────────────────────────────────────────────
    # skipna=True: days with some NaN timesteps still accumulate valid ones
    # n_timesteps: count of non-NaN values per day (QC diagnostic)
    daily_total = df.groupby("date")["flux"].sum(min_count=1)  # NaN if all NaN
    n_valid     = df.groupby("date")["flux"].count()

    df_daily = pd.DataFrame({
        "date"        : daily_total.index,
        "daily_total" : daily_total.values,
        "n_timesteps" : n_valid.values,
    }).reset_index(drop=True)

    return df_daily


# ============================================================
# CONVENIENCE WRAPPERS for common use cases
# ============================================================

def daily_GPP(
    subdaily_model,
    datetimes:    np.ndarray,
    timestep_sec: float,
) -> pd.DataFrame:
    """
    Daily accumulated GPP from pyrealm SubdailyPModel output.

    Converts pyrealm's native µg C m-2 s-1 → g C m-2 day-1.

    Parameters
    ----------
    subdaily_model : SubdailyPModel instance
    datetimes      : Subdaily datetime64 array
    timestep_sec   : Timestep duration in seconds (e.g. 1800 for 30-min)

    Returns
    -------
    pd.DataFrame with columns: date, daily_total (gC m-2 day-1), n_timesteps
    """
    return accumulate_daily(
        flux_subdaily = subdaily_model.gpp,
        datetimes     = datetimes,
        timestep_sec  = timestep_sec,
        flux_unit_in  = "ugC_m2_s",
        flux_unit_out = "gC_m2_day",
    )


def daily_Rd(
    Rd_subdaily:  np.ndarray,
    datetimes:    np.ndarray,
    timestep_sec: float,
) -> pd.DataFrame:
    """
    Daily accumulated leaf dark respiration.

    Converts µmol C m-2 s-1 → g C m-2 day-1.

    Parameters
    ----------
    Rd_subdaily  : Subdaily Rd array (µmol C m-2 s-1)
    datetimes    : Subdaily datetime64 array
    timestep_sec : Timestep duration in seconds
    """
    return accumulate_daily(
        flux_subdaily = Rd_subdaily,
        datetimes     = datetimes,
        timestep_sec  = timestep_sec,
        flux_unit_in  = "umol_C_m2_s",
        flux_unit_out = "gC_m2_day",
    )


def daily_net_carbon_gain(
    GPP_umolC:    np.ndarray,
    Rd_subdaily:  np.ndarray,
    datetimes:    np.ndarray,
    timestep_sec: float,
) -> pd.DataFrame:
    """
    Daily accumulated net carbon gain = GPP - Rd.

    Both inputs must be in µmol C m-2 s-1.
    Output is g C m-2 day-1.

    Parameters
    ----------
    GPP_umolC    : GPP in µmol C m-2 s-1 (= subdaily_model.gpp / 12.011)
    Rd_subdaily  : Rd in µmol C m-2 s-1
    datetimes    : Subdaily datetime64 array
    timestep_sec : Timestep duration in seconds
    """
    net_Ac = GPP_umolC - Rd_subdaily   # µmol C m-2 s-1
    return accumulate_daily(
        flux_subdaily = net_Ac,
        datetimes     = datetimes,
        timestep_sec  = timestep_sec,
        flux_unit_in  = "umol_C_m2_s",
        flux_unit_out = "gC_m2_day",
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================
# if __name__ == "__main__":

#     # Assume results dict from compute_leaf_respiration() pipeline
#     # and datetimes / timestep_sec are defined

#     timestep_sec = 1800.0   # 30-minute data

#     # ── Daily GPP ─────────────────────────────────────────────────────────────
#     df_GPP = daily_GPP(
#         subdaily_model = results["subdaily_model"],
#         datetimes      = datetimes,
#         timestep_sec   = timestep_sec,
#     )

#     # ── Daily Rd ──────────────────────────────────────────────────────────────
#     df_Rd = daily_Rd(
#         Rd_subdaily  = results["Rd_subdaily"],
#         datetimes    = datetimes,
#         timestep_sec = timestep_sec,
#     )

#     # ── Daily net carbon gain ─────────────────────────────────────────────────
#     df_net = daily_net_carbon_gain(
#         GPP_umolC    = results["GPP_umol"],
#         Rd_subdaily  = results["Rd_subdaily"],
#         datetimes    = datetimes,
#         timestep_sec = timestep_sec,
#     )

#     # ── Merge into single daily summary DataFrame ─────────────────────────────
#     df_daily_summary = pd.DataFrame({
#         "date"              : df_GPP["date"],
#         "GPP_gC_m2_day"     : df_GPP["daily_total"],
#         "Rd_gC_m2_day"      : df_Rd["daily_total"],
#         "net_Ac_gC_m2_day"  : df_net["daily_total"],
#         "n_timesteps"       : df_GPP["n_timesteps"],
#     })

#     print(df_daily_summary.head(20).to_string(index=False))
#     print(f"\nMean daily GPP:    {df_daily_summary['GPP_gC_m2_day'].mean():.4f} gC m-2 day-1")
#     print(f"Mean daily Rd:     {df_daily_summary['Rd_gC_m2_day'].mean():.4f} gC m-2 day-1")
#     print(f"Mean daily net Ac: {df_daily_summary['net_Ac_gC_m2_day'].mean():.4f} gC m-2 day-1")