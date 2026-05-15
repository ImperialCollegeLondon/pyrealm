"""
EEO Leaf Dark Respiration (H4) coupled with pyrealm SubdailyPModel
Following Ren et al. (2024) New Phytologist 241: 578-591

Confirmed design decisions:
  1. Acclimation window: ±30 min around noon (1 hour total)
  2. Vcmax25 acclimation: EWMA alpha=1/15 (pyrealm default)
  3. T_night_acc: 15-day simple rolling prior mean of nightly mean T
     Night-time defined as ppfd <= 0
  4. VPD: instantaneous subdaily values (no averaging)
  5. fr(T): pyrealm's calculate_ftemp_inst_rd (Heskel et al. 2016)
  6. Rd output: subdaily Rd = Rd25 * fr(tc) at each timestep
  7. Night-time mask: ppfd <= 0


"""

import numpy as np
import pandas as pd
from pyrealm.pmodel import (
    PModelEnvironment,
    SubdailyPModel,
    AcclimationModel,
)
from pyrealm.pmodel.functions import calc_ftemp_inst_rd


# ============================================================
# CONSTANTS  (Ren et al. 2024, Table 1)
# ============================================================
B      = 0.018   # EEO proportionality constant (Ren et al. 2024)
ALPHA  = 1 / 15  # EWMA alpha for Vcmax25 acclimation (Mengoli et al. 2022)
N_DAYS = 15      # Rolling window for T_night_acc (Ren et al. 2024, Fig. 2)


# ============================================================
# FUNCTION 1: Compute nightly mean temperature per day
# Night-time defined as ppfd <= 0 (Decision 7)
# ============================================================
def compute_nightly_mean_temperature(
    tc:        np.ndarray,
    ppfd:      np.ndarray,
    datetimes: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean night-time temperature for each calendar day.
    Night-time is defined as timesteps where ppfd <= 0.

    Parameters
    ----------
    tc        : Subdaily air temperature (°C), shape (n_timesteps,)
    ppfd      : Subdaily PPFD (µmol m-2 s-1), shape (n_timesteps,)
    datetimes : Subdaily datetime64 array, shape (n_timesteps,)

    Returns
    -------
    T_night_daily : Mean night-time temperature per day (°C),
                    shape (n_days,). NaN if no night-time data for a day.
    dates         : Unique calendar dates, shape (n_days,)

    Notes
    -----
    - Night-time = ppfd <= 0, consistent with Ren et al. (2024, p.581)
      who define night-time as sun elevation < 0 degrees.
    - Using ppfd <= 0 avoids external sun-angle libraries and is
      physically consistent since ppfd is already a model input.
    """
    # Convert to pandas for easy groupby operations
    df = pd.DataFrame({
        "tc"   : tc,
        "ppfd" : ppfd,
        "date" : pd.to_datetime(datetimes).normalize(),  # floor to day
    })

    # Mask to night-time only (ppfd <= 0)
    night_df = df[df["ppfd"] <= 0.0]

    # Mean night-time temperature per calendar day
    T_night_daily = (
        night_df.groupby("date")["tc"]
        .mean()
        .reindex(df["date"].unique())  # ensure all days present, NaN if missing
    )

    return T_night_daily.values, T_night_daily.index.values


# ============================================================
# FUNCTION 2: 15-day simple rolling prior mean
# For T_night acclimation (Ren et al. 2024 method)
# "T_night averaged over the prior nights" (p.581)
# ============================================================
def rolling_prior_mean(series: np.ndarray, N: int = N_DAYS) -> np.ndarray:
    """
    Simple arithmetic mean over the PRIOR N days.
    For day t: mean of days [t-N, t-1] (today excluded).

    First N entries are NaN (insufficient prior data).

    Parameters
    ----------
    series : Daily time series, shape (n_days,)
    N      : Window size in days (default = 15)

    Returns
    -------
    acc : Acclimated series, shape (n_days,). First N entries = NaN.

    Example (N=3, series=[10, 12, 14, 16, 18]):
        t=0,1,2 → NaN  (fewer than N prior days)
        t=3     → mean(10, 12, 14) = 12.0
        t=4     → mean(12, 14, 16) = 14.0
    """
    n   = len(series)
    acc = np.full(n, np.nan)
    for t in range(N, n):
        window = series[t - N : t]
        # Only compute mean if all values in window are valid
        if not np.any(np.isnan(window)):
            acc[t] = np.mean(window)
    return acc


# ============================================================
# FUNCTION 3: Compute Rd25 via H4 EEO hypothesis
# Ren et al. (2024) Table 1:
#   Rd,25 = b * Vcmax_opt / fr(T_night_acc)
# where fr is Heskel et al. (2016) evaluated at T_night_acc
# Note: applying fr to acclimated T is an approximation —
# Heskel et al. (2016) derived fr from instantaneous ramps,
# but this is how Ren et al. (2024) use it throughout.
# ============================================================
def compute_Rd25_H4(
    Vcmax_opt:   np.ndarray,
    T_night_acc: np.ndarray,
    b:           float = B,
) -> np.ndarray:
    """
    Compute Rd,25 following H4 (EEO hypothesis) of Ren et al. (2024).

    Rd,25 = b * Vcmax_opt / fr(T_night_acc)

    Uses pyrealm's calculate_ftemp_inst_rd for fr (Heskel et al. 2016).

    Parameters
    ----------
    Vcmax_opt    : Daily Vcmax at acclimated noon temperature (µmol m-2 s-1)
                   = pmodel_acclim.vcmax from SubdailyPModel
    T_night_acc  : 15-day prior rolling mean of nightly mean T (°C)
    b            : EEO proportionality constant (default = 0.018)

    Returns
    -------
    Rd25 : Daily dark respiration at 25°C (µmol m-2 s-1)
           NaN where T_night_acc is NaN (first 15 days)
    """
    fr_night = calc_ftemp_inst_rd(T_night_acc)  # pyrealm Heskel function
    Rd25     = b * Vcmax_opt / fr_night
    return Rd25


# ============================================================
# MAIN PIPELINE
# ============================================================
def compute_leaf_respiration(
    datetimes:      np.ndarray,
    tc_subdaily:    np.ndarray,
    vpd_subdaily:   np.ndarray,
    co2_subdaily:   np.ndarray,
    patm_subdaily:  np.ndarray,
    fapar_subdaily: np.ndarray,
    ppfd_subdaily:  np.ndarray,
    alpha:          float = ALPHA,
    b:              float = B,
    N_night:        int   = N_DAYS,
    method_optchi:  str   = "prentice14",
    method_kphio:   str   = "temperature",
    method_jmaxlim: str   = "wang17",
) -> dict:
    """
    Full pipeline: SubdailyPModel (pyrealm) + H4 leaf dark respiration.

    Outputs Rd at every subdaily timestep for net carbon gain calculation:
        Net carbon gain = GPP - Rd  (both in µmol C m-2 s-1)

    Parameters
    ----------
    datetimes      : Subdaily datetime64 array, shape (n_timesteps,)
    tc_subdaily    : Air temperature (°C), shape (n_timesteps,)
    vpd_subdaily   : VPD (Pa), instantaneous, shape (n_timesteps,)
    co2_subdaily   : CO2 concentration (ppm), shape (n_timesteps,)
    patm_subdaily  : Atmospheric pressure (Pa), shape (n_timesteps,)
    fapar_subdaily : fAPAR (-), shape (n_timesteps,)
    ppfd_subdaily  : PPFD (µmol m-2 s-1), shape (n_timesteps,)
                     Also used as night-time mask (ppfd <= 0 = night)
    alpha          : EWMA alpha for Vcmax25 acclimation (default=1/15)
    b              : H4 EEO proportionality constant (default=0.018)
    N_night        : Rolling window for T_night_acc in days (default=15)
    method_optchi  : pyrealm optimal chi method (default='prentice14')
    method_kphio   : pyrealm quantum yield method (default='temperature')
    method_jmaxlim : pyrealm Jmax limitation (default='wang17')

    Returns
    -------
    dict with keys:
        'GPP_subdaily'    : Gross primary production (µg C m-2 s-1)
        'Vcmax_opt_daily' : Vcmax at noon acclim. conditions (µmol m-2 s-1)
        'Vcmax25_daily'   : EWMA-acclimated Vcmax25 (µmol m-2 s-1)
        'T_night_daily'   : Nightly mean temperature per day (°C)
        'T_night_acc'     : 15-day rolling prior mean T_night (°C)
        'Rd25_daily'      : Rd at 25°C from H4 (µmol m-2 s-1)
        'Rd25_subdaily'   : Rd25 filled to subdaily scale (µmol m-2 s-1)
        'Rd_subdaily'     : Rd at actual tc (µmol m-2 s-1) — primary output
        'net_Ac_subdaily' : Net carbon gain = GPP/k_c - Rd (µmol C m-2 s-1)
        'dates'           : Calendar dates for daily outputs
        'subdaily_model'  : SubdailyPModel instance (full pyrealm output)
        'acclim_model'    : AcclimationModel instance
    """

    # ── STEP 1: Build PModelEnvironment ──────────────────────────────────────
    # VPD is passed as instantaneous subdaily values — no averaging (Decision 4)
    print("Step 1: Building PModelEnvironment...")
    env = PModelEnvironment(
        tc    = tc_subdaily,
        vpd   = vpd_subdaily,
        co2   = co2_subdaily,
        patm  = patm_subdaily,
        fapar = fapar_subdaily,
        ppfd  = ppfd_subdaily,
    )

    # ── STEP 2: AcclimationModel — ±30 min around noon (Decision 1) ──────────
    # This defines the daily window for Vcmax_opt acclimation
    # EWMA alpha=1/15 implements ~15-day memory (Decision 2)
    print("Step 2: Setting up AcclimationModel (noon ±30 min, α=1/15)...")
    acclim_model = AcclimationModel(
        datetimes      = datetimes,
        alpha          = alpha,
        allow_holdover = True,
    )
    acclim_model.set_window(
        window_center = np.timedelta64(12, "h"),
        half_width    = np.timedelta64(30, "m"),
    )

    # ── STEP 3: Fit SubdailyPModel ────────────────────────────────────────────
    # Internally:
    #   a) Extracts daily mean conditions in noon window → noon acclim. env.
    #   b) Runs standard PModel on daily noon means → Vcmax_opt
    #   c) Normalises to 25°C → Vcmax25_optimal
    #   d) Applies EWMA (α=1/15) → Vcmax25_realised
    #   e) Fills back to subdaily + Arrhenius scaling → subdaily Vcmax
    print("Step 3: Fitting SubdailyPModel...")
    subdaily_model = SubdailyPModel(
        env            = env,
        acclim_model   = acclim_model,
        method_optchi  = method_optchi,
        method_kphio   = method_kphio,
        method_jmaxlim = method_jmaxlim,
    )

    # ── STEP 4: Extract daily Vcmax_opt ──────────────────────────────────────
    # pmodel_acclim.vcmax = Vcmax at acclimated noon temperature (= Vcmax_opt)
    # vcmax25_daily_realised = EWMA-acclimated Vcmax25
    print("Step 4: Extracting daily Vcmax_opt and Vcmax25...")
    Vcmax_opt_daily = subdaily_model.pmodel_acclim.vcmax
    Vcmax25_daily   = subdaily_model.vcmax25_daily_realised

    # ── STEP 5: Compute nightly mean T and 15-day rolling prior mean ──────────
    # Night-time = ppfd <= 0 (Decision 3 & 7)
    # 15-day simple rolling prior mean (not EWMA) — Ren et al. 2024 method
    print("Step 5: Computing T_night_acc (15-day rolling prior mean)...")
    T_night_daily, dates = compute_nightly_mean_temperature(
        tc        = tc_subdaily,
        ppfd      = ppfd_subdaily,
        datetimes = datetimes,
    )
    T_night_acc = rolling_prior_mean(T_night_daily, N=N_night)

    # ── STEP 6: Compute Rd25 via H4 EEO ──────────────────────────────────────
    # Rd,25 = b * Vcmax_opt / fr(T_night_acc)
    # fr = Heskel et al. (2016) via pyrealm's calculate_ftemp_inst_rd
    # Note: first N_night days will be NaN (insufficient prior T_night data)
    print("Step 6: Computing Rd25 via H4 EEO hypothesis...")
    Rd25_daily = compute_Rd25_H4(
        Vcmax_opt   = Vcmax_opt_daily,
        T_night_acc = T_night_acc,
        b           = b,
    )

    # ── STEP 7: Fill Rd25 to subdaily scale ──────────────────────────────────
    # Use same fill method as pyrealm for consistency with Vcmax25 filling
    print("Step 7: Filling Rd25 to subdaily scale...")
    Rd25_subdaily = acclim_model.fill_daily_to_subdaily(Rd25_daily)

    # ── STEP 8: Scale Rd25 to actual subdaily temperature ────────────────────
    # Rd(T) = Rd25 * fr(tc)
    # Uses pyrealm's calculate_ftemp_inst_rd — this is the INSTANTANEOUS
    # temperature response (Heskel et al. 2016), applied to actual tc.
    # This is the valid use of fr (instantaneous kinetics, not acclimation).
    print("Step 8: Scaling Rd25 to actual subdaily temperature...")
    fr_subdaily  = calc_ftemp_inst_rd(tc_subdaily)
    Rd_subdaily  = Rd25_subdaily * fr_subdaily

    # ── STEP 9: Net carbon gain at each subdaily timestep ────────────────────
    # GPP from pyrealm is in µg C m-2 s-1; convert to µmol C m-2 s-1
    # using molecular mass of C (12.011 g mol-1)
    # Rd is in µmol C m-2 s-1
    # Net Ac = GPP (µmol C) - Rd (µmol C) — both per m2 leaf area per s
    print("Step 9: Computing net carbon gain (GPP - Rd)...")
    k_c_molmass   = subdaily_model.env.core_const.k_c_molmass  # g mol-1
    GPP_umol      = subdaily_model.gpp / k_c_molmass            # µmol C m-2 s-1
    net_Ac_subdaily = GPP_umol - Rd_subdaily

    print("Done!")
    return {
        "GPP_subdaily"    : subdaily_model.gpp,     # µg C m-2 s-1
        "GPP_umol"        : GPP_umol,               # µmol C m-2 s-1
        "Vcmax_opt_daily" : Vcmax_opt_daily,         # µmol m-2 s-1
        "Vcmax25_daily"   : Vcmax25_daily,           # µmol m-2 s-1
        "T_night_daily"   : T_night_daily,           # °C, per day
        "T_night_acc"     : T_night_acc,             # °C, 15-day rolling mean
        "Rd25_daily"      : Rd25_daily,              # µmol m-2 s-1
        "Rd25_subdaily"   : Rd25_subdaily,           # µmol m-2 s-1
        "Rd_subdaily"     : Rd_subdaily,             # µmol C m-2 s-1
        "net_Ac_subdaily" : net_Ac_subdaily,         # µmol C m-2 s-1
        "dates"           : dates,                   # calendar dates (daily)
        "subdaily_model"  : subdaily_model,
        "acclim_model"    : acclim_model,
    }


# ============================================================
# EXAMPLE USAGE with synthetic half-hourly data
# ============================================================
if __name__ == "__main__":

    # ── Generate synthetic half-hourly data (60 days) ────────────────────────
    n_days   = 60
    freq_min = 30
    n_steps  = n_days * 24 * (60 // freq_min)

    datetimes = pd.date_range(
        "2020-06-01", periods=n_steps, freq=f"{freq_min}min"
    ).to_numpy()

    # Hour of day (0.0 to 23.5)
    hod = (np.arange(n_steps) % (24 * 60 // freq_min)) / (60 // freq_min)
    doy = np.arange(n_steps) / (24 * 60 // freq_min)

    # Temperature: seasonal trend + diurnal cycle
    T_mean      = 15.0 + 8.0 * np.sin(2 * np.pi * doy / 365)
    tc_subdaily = T_mean + 7.0 * np.sin(2 * np.pi * (hod - 6) / 24)

    # PPFD: zero at night, sinusoidal during day (06:00-18:00)
    ppfd_subdaily = np.where(
        (hod >= 6) & (hod <= 18),
        1200 * np.sin(np.pi * (hod - 6) / 12),
        0.0
    )

    # Other forcing
    vpd_subdaily  = np.clip(
        300 + 600 * np.sin(2 * np.pi * (hod - 6) / 24), 50, np.inf
    )
    co2_subdaily  = np.full(n_steps, 415.0)
    patm_subdaily = np.full(n_steps, 101325.0)
    fapar_subdaily = np.full(n_steps, 0.7)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    results = compute_leaf_respiration(
        datetimes      = datetimes,
        tc_subdaily    = tc_subdaily,
        vpd_subdaily   = vpd_subdaily,
        co2_subdaily   = co2_subdaily,
        patm_subdaily  = patm_subdaily,
        fapar_subdaily = fapar_subdaily,
        ppfd_subdaily  = ppfd_subdaily,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY (daytime only, ppfd > 0)")
    print("=" * 55)
    daytime_mask = ppfd_subdaily > 0

    print(f"GPP (mean daytime):      "
          f"{np.nanmean(results['GPP_umol'][daytime_mask]):.3f} µmol C m-2 s-1")
    print(f"Rd  (mean daytime):      "
          f"{np.nanmean(results['Rd_subdaily'][daytime_mask]):.4f} µmol C m-2 s-1")
    print(f"Net Ac (mean daytime):   "
          f"{np.nanmean(results['net_Ac_subdaily'][daytime_mask]):.3f} µmol C m-2 s-1")
    print(f"Rd25 (daily mean):       "
          f"{np.nanmean(results['Rd25_daily']):.4f} µmol C m-2 s-1")
    print(f"T_night_acc (mean):      "
          f"{np.nanmean(results['T_night_acc']):.2f} °C")
    print(f"Vcmax_opt (daily mean):  "
          f"{np.nanmean(results['Vcmax_opt_daily']):.3f} µmol C m-2 s-1")
    print(f"Vcmax25 (daily mean):    "
          f"{np.nanmean(results['Vcmax25_daily']):.3f} µmol C m-2 s-1")