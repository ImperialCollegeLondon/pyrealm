import numpy as np
import pandas as pd
from pyrealm.pmodel import PModelEnvironment, PModel, SubdailyPModel
from pyrealm.pmodel.acclimation import AcclimationModel
from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
from pyrealm.constants import PModelConst

# ── Import accumulation functions from local module ───────────────────────────
from .accumulate_daily import daily_GPP


# ============================================================
# HELPER: ARRHENIUS FACTOR AT DAILY MEAN TEMPERATURE
# ============================================================

def _compute_arrhenius_factor(
    tc:     np.ndarray,
    Ha:     float,
    R:      float = 8.314,
    Tref_K: float = 298.15,
) -> np.ndarray:
    """
    Compute the Arrhenius temperature scaling factor h_T:

        h_T = exp( Ha/R * (1/Tref - 1/T) )

    where T and Tref are in Kelvin.

    Parameters
    ----------
    tc     : np.ndarray — temperature in °C
    Ha     : float      — activation energy in J mol-1
                          (use pmodel_const.arrhenius_vcmax.Ha)
    R      : float      — universal gas constant, J mol-1 K-1
    Tref_K : float      — reference temperature in Kelvin (25°C = 298.15 K)

    Returns
    -------
    h_T : np.ndarray — dimensionless Arrhenius scaling factor,
                       shape same as tc
    """
    T_K = tc + 273.15
    return np.exp(Ha / R * (1.0 / Tref_K - 1.0 / T_K))


# ============================================================
# HELPER: GROWING SEASON FRACTION f
# ============================================================

def _compute_growing_season(
    tc_subdaily: np.ndarray,
    datetimes:   np.ndarray,
    threshold_C: float = 0.0,
) -> dict:
    """
    Compute growing season diagnostics from subdaily temperature.

    Growing season defined as per Wang et al. (2023):
        days where daily mean temperature > threshold_C (default 0°C)

    Parameters
    ----------
    tc_subdaily  : np.ndarray — subdaily temperature in °C
    datetimes    : np.ndarray of datetime64
    threshold_C  : float — temperature threshold for growing season
                           (default 0°C)

    Returns
    -------
    dict with keys:
        'f'             : float            — fraction of year in growing season
        'leaf_out_doy'  : int              — first DOY where T_mean > threshold
        'LL_de'         : float            — deciduous leaf longevity (days)
        'tc_daily_mean' : np.ndarray       — daily mean temperature (n_days,)
        'dates_daily'   : pd.DatetimeIndex — one date per day
    """
    df = pd.DataFrame({
        "date" : pd.to_datetime(datetimes).normalize(),
        "tc"   : tc_subdaily,
    })

    # Daily mean temperature over full 24h
    tc_daily    = df.groupby("date")["tc"].mean()
    dates_daily = pd.DatetimeIndex(tc_daily.index)

    # Growing season mask: days above temperature threshold
    gs_mask = tc_daily.values > threshold_C

    # f: fraction of days in growing season
    f = float(gs_mask.mean())

    # leaf_out_doy: first DOY where growing season begins
    gs_indices = np.where(gs_mask)[0]
    if len(gs_indices) == 0:
        raise ValueError(
            "No growing season days found "
            f"(no days with T_mean > {threshold_C}°C). "
            "Check temperature inputs."
        )
    leaf_out_doy = int(dates_daily[gs_indices[0]].day_of_year)

    # Deciduous leaf longevity = growing season length in days
    LL_de = 365.0 * f

    return {
        "f"             : f,
        "leaf_out_doy"  : leaf_out_doy,
        "LL_de"         : LL_de,
        "tc_daily_mean" : tc_daily.values,
        "dates_daily"   : dates_daily,
    }


# ============================================================
# HELPER: DAILY MEAN mc (Generation 2 consistent)
# ============================================================

def _compute_mc_daily(
    subdaily_env:  PModelEnvironment,
    pmodel_acclim: PModel,
    acclim_model:  AcclimationModel,
    pmodel_const:  PModelConst,
) -> np.ndarray:
    """
    Compute daily mean m_c = (ci - gammastar) / (ci + kmm)
    using Generation 2 consistent ci (via xi_real).

    Pipeline:
        xi_acclim  →  apply_acclimation()       →  xi_real
        xi_real    →  fill_daily_to_subdaily()   →  xi_subdaily
        xi_subdaily + subdaily_env  →  OptimalChiPrentice14  →  ci_subdaily
        mc_subdaily  →  get_daily_means()        →  mc_daily

    Parameters
    ----------
    subdaily_env  : PModelEnvironment — full subdaily environment
    pmodel_acclim : PModel            — P model fitted on acclimation window
    acclim_model  : AcclimationModel  — acclimation model instance
    pmodel_const  : PModelConst       — P model constants

    Returns
    -------
    mc_daily : np.ndarray, shape (n_days,)
    """
    # Step 1: xi_acclim — instantaneously optimal (Generation 1)
    xi_acclim = pmodel_acclim.optchi.xi                           # (n_days,)

    # Step 2: xi_real — memory-smoothed (Generation 2)
    xi_real = acclim_model.apply_acclimation(xi_acclim)           # (n_days,)

    # Step 3: xi_subdaily — broadcast back to subdaily timesteps
    xi_subdaily = acclim_model.fill_daily_to_subdaily(xi_real)    # (n_subdaily,)

    # Step 4: ci_subdaily via OptimalChiPrentice14 with xi_subdaily
    optchi_subdaily = OptimalChiPrentice14(
        env          = subdaily_env,
        pmodel_const = pmodel_const,
    )
    optchi_subdaily.estimate_chi(xi_values=xi_subdaily)
    ci_subdaily = optchi_subdaily.ci                              # (n_subdaily,)

    # Step 5: mc_subdaily using fast-response gammastar and kmm
    mc_subdaily = (
        (ci_subdaily - subdaily_env.gammastar) /
        (ci_subdaily + subdaily_env.kmm)
    )                                                             # (n_subdaily,)

    # Step 6: aggregate to daily mean
    mc_daily = acclim_model.get_daily_means(mc_subdaily)          # (n_days,)

    return mc_daily


# ============================================================
# HELPER: DAILY REALISED Vcmax25 (Generation 2)
# ============================================================

def _compute_vcmax25_real(
    pmodel_acclim: PModel,
    acclim_model:  AcclimationModel,
) -> np.ndarray:
    """
    Compute daily realised Vcmax25 (Generation 2):

        vcmax25_acclim  →  apply_acclimation()  →  vcmax25_real

    Parameters
    ----------
    pmodel_acclim : PModel           — P model fitted on acclimation window
    acclim_model  : AcclimationModel — acclimation model instance

    Returns
    -------
    vcmax25_real : np.ndarray, shape (n_days,) — µmol m-2 s-1
    """
    vcmax25_acclim = pmodel_acclim.vcmax25                           # (n_days,)
    vcmax25_real   = acclim_model.apply_acclimation(vcmax25_acclim)  # (n_days,)
    return vcmax25_real


# ============================================================
# HELPER: DECIDUOUS LEAF AGE SCALAR
# ============================================================

def _scalar_deciduous(
    dates_daily:  pd.DatetimeIndex,
    leaf_out_doy: int,
    LL_de:        float,
) -> tuple:
    """
    Compute daily leaf age scalar for deciduous species:

        scalar_de(t) = 1 - t / LL_de

    where t = DOY - leaf_out_doy, clipped to [0, LL_de].

    Outside the growing season the scalar is set to 1.0 since
    GPP is already ~0 from the P model (no double-correction needed).

    Parameters
    ----------
    dates_daily  : pd.DatetimeIndex — one entry per day
    leaf_out_doy : int              — DOY of leaf flush
    LL_de        : float            — deciduous leaf longevity (days)

    Returns
    -------
    scalar_de  : np.ndarray, shape (n_days,) — leaf age scalar
    leaf_age_t : np.ndarray, shape (n_days,) — leaf age in days
                 (NaN outside growing season)
    """
    doy = dates_daily.day_of_year.values.astype(float)

    # Leaf age in days since flush, clipped to growing season
    leaf_age_t = np.clip(doy - leaf_out_doy, 0.0, LL_de)

    # Linear decline scalar: 1 at flush → 0 at end of growing season
    scalar_de = 1.0 - leaf_age_t / LL_de

    # Outside growing season: no correction (GPP already ~0)
    outside_gs = (doy < leaf_out_doy) | (doy > leaf_out_doy + LL_de)
    scalar_de[outside_gs]  = 1.0
    leaf_age_t[outside_gs] = np.nan

    return scalar_de, leaf_age_t


# ============================================================
# HELPER: EVERGREEN CANOPY-MEAN LEAF AGE SCALAR
# ============================================================

def _scalar_evergreen(
    f:          float,
    h_T_daily:  np.ndarray,
    mc_daily:   np.ndarray,
    Cc:         float,
    u:          float,
) -> tuple:
    """
    Compute daily canopy-mean leaf age scalar for evergreen species
    using Approach A — uniform cohort age distribution
    (Wang et al. 2023, Eq. 5):

        LL_ev / b  = sqrt( 2 * Cc * f / (u * h_T * mc) )
        scalar_ev  = 1 - (LL_ev / b) / 2

    Parameters
    ----------
    f          : float      — growing season fraction (dimensionless)
    h_T_daily  : np.ndarray — Arrhenius factor at daily mean T, (n_days,)
    mc_daily   : np.ndarray — daily mean CO2 limitation factor,  (n_days,)
    Cc         : float      — construction cost (gC gC-1)
    u          : float      — aging rate constant (dimensionless)

    Returns
    -------
    scalar_ev  : np.ndarray, shape (n_days,) — canopy-mean age scalar
    LL_b_ratio : np.ndarray, shape (n_days,) — LL_ev / b (diagnostic)
    """
    # LL_ev / b — fully determined by environmental conditions
    LL_b_ratio = np.sqrt(
        (2.0 * Cc * f) / (u * h_T_daily * mc_daily)
    )

    # Canopy-mean scalar under uniform age distribution
    scalar_ev = 1.0 - LL_b_ratio / 2.0

    # Clip to [0, 1]: LL_b_ratio > 2 is physically unrealistic
    scalar_ev = np.clip(scalar_ev, 0.0, 1.0)

    return scalar_ev, LL_b_ratio


# ============================================================
# MAIN FUNCTION
# ============================================================

def compute_gpp_leaf_age(
    subdaily_model:    SubdailyPModel,
    subdaily_env:      PModelEnvironment,
    pmodel_acclim:     PModel,
    acclim_model:      AcclimationModel,
    tc_subdaily:       np.ndarray,
    datetime_subdaily: np.ndarray,
    timestep_sec:      float,
    plant_type:        str,
    pmodel_const:      PModelConst = None,
    Cc:                float       = 13.23,
    u:                 float       = 768.0,
    threshold_C:       float       = 0.0,
) -> pd.DataFrame:
    """
    Compute daily GPP with and without leaf age correction following
    Wang et al. (2023, Science Advances, doi:10.1126/sciadv.add5667).

    The leaf age effect is a multiplicative scalar on P model GPP:

        GPP_aged = GPP_ori × scalar(t)

    Deciduous (linear decline over growing season, Eq. 4):
        scalar_de(t) = 1 - t / LL_de
        t      = days since leaf flush (DOY - leaf_out_doy)
        LL_de  = 365 * f

    Evergreen (uniform cohort age distribution, Eq. 5):
        scalar_ev = 1 - (1/2) * sqrt( 2*Cc*f / (u * h_T * mc) )

    All internal variables use Generation 2 (memory-smoothed, daily
    realised) values, consistent with SubdailyPModel GPP output:
        Vcmax25 : vcmax25_real  via apply_acclimation()
        mc      : from ci_subdaily (using xi_real) → daily mean
        h_T     : Arrhenius factor at full 24h daily mean temperature

    Parameters
    ----------
    subdaily_model    : SubdailyPModel
        Fitted pyrealm SubdailyPModel instance.
    subdaily_env      : PModelEnvironment
        Full subdaily PModelEnvironment (for gammastar, kmm, ci).
    pmodel_acclim     : PModel
        PModel fitted on the acclimation window (daily).
        Provides vcmax25_acclim and optchi.xi.
    acclim_model      : AcclimationModel
        Provides apply_acclimation(), fill_daily_to_subdaily(),
        get_daily_means(), observation_dates.
    tc_subdaily       : np.ndarray, shape (n_subdaily,)
        Subdaily air temperature in °C.
    datetime_subdaily : np.ndarray of datetime64, shape (n_subdaily,)
        Subdaily datetime array.
    timestep_sec      : float
        Duration of each subdaily timestep in seconds (e.g. 1800.0).
    plant_type        : str
        One of "deciduous", "evergreen", or "both".
    pmodel_const      : PModelConst, optional
        P model constants instance. Uses default if None.
    Cc                : float, optional
        Leaf construction cost (gC gC-1). Default 13.23.
    u                 : float, optional
        Leaf aging rate constant (dimensionless). Default 768.0.
    threshold_C       : float, optional
        Temperature threshold for growing season (°C). Default 0.0.

    Returns
    -------
    df_out : pd.DataFrame
        Daily output with columns:

        Always present:
            date          : calendar date
            GPP_ori       : daily GPP, no leaf age correction (gC m-2 day-1)
            f             : growing season fraction
            h_T_daily     : Arrhenius temperature scaling factor
            mc_daily      : daily mean CO2 limitation factor
            vcmax25_real  : daily realised Vcmax25 (µmol m-2 s-1)
            n_timesteps   : valid subdaily timesteps per day (QC)

        If plant_type in ("deciduous", "both"):
            GPP_aged_de   : GPP with deciduous correction (gC m-2 day-1)
            scalar_de     : deciduous age scalar
            leaf_age_t    : leaf age in days (NaN outside growing season)

        If plant_type in ("evergreen", "both"):
            GPP_aged_ev   : GPP with evergreen correction (gC m-2 day-1)
            scalar_ev     : evergreen canopy-mean age scalar
            LL_b_ratio    : LL_ev / b ratio (diagnostic)

    Raises
    ------
    ValueError : if plant_type is not recognised
    ValueError : if no growing season days are found
    UserWarning: if scalar <= 0 for > 5% of days (deciduous)
    UserWarning: if LL_b_ratio > 2 for any days (evergreen)

    Examples
    --------
    df_result = compute_gpp_leaf_age(
        subdaily_model    = subdaily_model,
        subdaily_env      = subdaily_env,
        pmodel_acclim     = pmodel_acclim,
        acclim_model      = acclim_model,
        tc_subdaily       = tc_subdaily,
        datetime_subdaily = datetimes,
        timestep_sec      = 1800.0,
        plant_type        = "both",
    )
    """
    import warnings

    # ── Input validation ──────────────────────────────────────────────────────
    valid_types = ("deciduous", "evergreen", "both")
    if plant_type not in valid_types:
        raise ValueError(
            f"plant_type must be one of {valid_types}, got '{plant_type}'."
        )
    if pmodel_const is None:
        pmodel_const = PModelConst()

    # ── BLOCK 1: Daily GPP_ori ────────────────────────────────────────────────
    # Reuses daily_GPP() from accumulate_daily.py
    # µg C m-2 s-1 → gC m-2 day-1
    df_gpp      = daily_GPP(
        subdaily_model = subdaily_model,
        datetimes      = datetime_subdaily,
        timestep_sec   = timestep_sec,
    )
    GPP_ori     = df_gpp["daily_total"].values    # (n_days,)
    dates_daily = pd.DatetimeIndex(df_gpp["date"].values)
    n_timesteps = df_gpp["n_timesteps"].values

    # ── BLOCK 2: Growing season diagnostics ──────────────────────────────────
    gs           = _compute_growing_season(
        tc_subdaily = tc_subdaily,
        datetimes   = datetime_subdaily,
        threshold_C = threshold_C,
    )
    f             = gs["f"]
    leaf_out_doy  = gs["leaf_out_doy"]
    LL_de         = gs["LL_de"]
    tc_daily_mean = gs["tc_daily_mean"]           # (n_days,) full 24h mean

    # ── BLOCK 3: Daily realised Vcmax25 (Generation 2) ───────────────────────
    vcmax25_real = _compute_vcmax25_real(
        pmodel_acclim = pmodel_acclim,
        acclim_model  = acclim_model,
    )                                             # (n_days,) µmol m-2 s-1

    # ── BLOCK 4: Daily mean mc (Generation 2 consistent) ─────────────────────
    mc_daily = _compute_mc_daily(
        subdaily_env  = subdaily_env,
        pmodel_acclim = pmodel_acclim,
        acclim_model  = acclim_model,
        pmodel_const  = pmodel_const,
    )                                             # (n_days,)

    # Sanity check: mc must be positive
    if np.any(mc_daily <= 0):
        n_bad = int(np.sum(mc_daily <= 0))
        raise ValueError(
            f"mc_daily has {n_bad} non-positive values. "
            "Check ci, gammastar, and kmm inputs."
        )

    # ── BLOCK 5: h_T at full 24h daily mean temperature ──────────────────────
    # Consistent with vcmax25_real as a daily-integrated trait (Generation 2)
    Ha        = pmodel_const.arrhenius_vcmax.Ha   # activation energy, J mol-1
    h_T_daily = _compute_arrhenius_factor(
        tc = tc_daily_mean,
        Ha = Ha,
    )                                             # (n_days,)

    # ── BLOCK 6: Assemble base output DataFrame ───────────────────────────────
    df_out = pd.DataFrame({
        "date"         : dates_daily,
        "GPP_ori"      : GPP_ori,
        "f"            : f,
        "h_T_daily"    : h_T_daily,
        "mc_daily"     : mc_daily,
        "vcmax25_real" : vcmax25_real,
        "n_timesteps"  : n_timesteps,
    })

    # ── BLOCK 7: Deciduous leaf age correction ────────────────────────────────
    if plant_type in ("deciduous", "both"):
        scalar_de, leaf_age_t = _scalar_deciduous(
            dates_daily  = dates_daily,
            leaf_out_doy = leaf_out_doy,
            LL_de        = LL_de,
        )

        n_zero = int(np.sum(scalar_de <= 0))
        if n_zero / len(scalar_de) > 0.05:
            warnings.warn(
                f"{n_zero} days ({100 * n_zero / len(scalar_de):.1f}%) have "
                "deciduous scalar <= 0. Check leaf_out_doy and LL_de.",
                UserWarning,
            )

        df_out["scalar_de"]   = scalar_de
        df_out["leaf_age_t"]  = leaf_age_t
        df_out["GPP_aged_de"] = GPP_ori * scalar_de

    # ── BLOCK 8: Evergreen leaf age correction ────────────────────────────────
    if plant_type in ("evergreen", "both"):
        scalar_ev, LL_b_ratio = _scalar_evergreen(
            f         = f,
            h_T_daily = h_T_daily,
            mc_daily  = mc_daily,
            Cc        = Cc,
            u         = u,
        )

        n_clipped = int(np.sum(LL_b_ratio > 2.0))
        if n_clipped > 0:
            warnings.warn(
                f"{n_clipped} days have LL_b_ratio > 2 (scalar clipped to 0). "
                "This may indicate extreme environmental conditions.",
                UserWarning,
            )

        df_out["scalar_ev"]   = scalar_ev
        df_out["LL_b_ratio"]  = LL_b_ratio
        df_out["GPP_aged_ev"] = GPP_ori * scalar_ev

    return df_out


# ============================================================
# EXAMPLE USAGE
# ============================================================
# if __name__ == "__main__":
#
#     # Folder structure assumed:
#     #   your_project/
#     #   ├── accumulate_daily.py   ← existing accumulation functions
#     #   └── gpp_leaf_age.py       ← this script
#
#     # ── Objects already available from your P model pipeline ─────────────
#     # subdaily_model    : SubdailyPModel
#     # subdaily_env      : PModelEnvironment (subdaily)
#     # pmodel_acclim     : PModel (acclimation window)
#     # acclim_model      : AcclimationModel
#     # tc_subdaily       : np.ndarray (°C)
#     # datetimes         : np.ndarray of datetime64
#     # timestep_sec      : float (e.g. 1800.0)
#
#     df_result = compute_gpp_leaf_age(
#         subdaily_model    = subdaily_model,
#         subdaily_env      = subdaily_env,
#         pmodel_acclim     = pmodel_acclim,
#         acclim_model      = acclim_model,
#         tc_subdaily       = tc_subdaily,
#         datetime_subdaily = datetimes,
#         timestep_sec      = 1800.0,
#         plant_type        = "both",
#         Cc                = 13.23,
#         u                 = 768.0,
#         threshold_C       = 0.0,
#     )
#
#     print(df_result.head(10).to_string(index=False))
#     print(f"\nGrowing season fraction f       : {df_result['f'].iloc[0]:.3f}")
#     print(f"Mean GPP_ori                    : {df_result['GPP_ori'].mean():.4f} gC m-2 day-1")
#     print(f"Mean GPP_aged_de (deciduous)    : {df_result['GPP_aged_de'].mean():.4f} gC m-2 day-1")
#     print(f"Mean GPP_aged_ev (evergreen)    : {df_result['GPP_aged_ev'].mean():.4f} gC m-2 day-1")
#     print(f"Mean deciduous scalar           : {df_result['scalar_de'].mean():.4f}")
#     print(f"Mean evergreen scalar           : {df_result['scalar_ev'].mean():.4f}")
#     print(f"Mean LL/b ratio (evergreen)     : {df_result['LL_b_ratio'].mean():.4f}")