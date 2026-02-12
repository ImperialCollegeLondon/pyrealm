"""Production-limited (PL) phenology seasonal simulator (Python port).

It computes steady-state LAI, applies a moisture-driven
lag, and converts the simulated canopy structure back to fAPAR.


Notes added David Orme (2026-01-19):

* This is the reference implementation for Ziqi Zhu's phenology method - it does not
  follow the pyrealm standards for code.

* I have also added the original implementation of the function used to calculate
  annual fapar max to keep the reference implementation in a single file. This
  function was untyped and so has been decorated with no_type_check to suppress
  warnings.

* I have also added some extra code to the _apply_lag function that simply tracks the
  internal variables used in the lagging process. This was used in implementing the
  method to work out an efficient approach to vectorising the lagging calculations. The
  function contains commented out code used to write out a CSV of the full internal
  lag calculations.
"""

from __future__ import annotations

from typing import no_type_check

import numpy as np


# ziqi's function
@no_type_check
def cal_fapar(fapar_carbon, fapar_water, options=None):
    """Function to calculate maximum fapar from water and energy limited versions."""
    # Handle options
    if options is None:
        options = {}

    const_budyko = options.get("const_budyko", 4)

    # Convert to float arrays
    fapar_carbon = np.asarray(fapar_carbon, dtype=float)
    fapar_water = np.asarray(fapar_water, dtype=float)

    # Handle scalar broadcasting
    if fapar_carbon.ndim == 0 and fapar_water.ndim > 0:
        fapar_carbon = np.full_like(fapar_water, fapar_carbon)
    elif fapar_water.ndim == 0 and fapar_carbon.ndim > 0:
        fapar_water = np.full_like(fapar_carbon, fapar_water)
    elif fapar_carbon.shape != fapar_water.shape:
        try:
            # Test if broadcasting works
            _ = fapar_carbon + fapar_water
        except ValueError:
            raise ValueError(
                "fapar_carbon and fapar_water must be scalar or equal-sized arrays"
            )

    # Store original shape
    sza = fapar_carbon.shape

    # Flatten arrays
    flatC = fapar_carbon.ravel()
    flatW = fapar_water.ravel()
    n = flatC.size

    # Initialize output arrays
    flat_out = np.full(n, np.nan)
    flat_ratio = np.full(n, np.nan)
    flat_factor = np.full(n, np.nan)

    safety_eps = np.finfo(float).eps

    # Loop through each element
    for ii in range(n):
        fc = flatC[ii]
        fw = flatW[ii]

        # Safe denominator
        denom_safe = fw + (fw == 0) * safety_eps
        r_local = fc / denom_safe
        flat_ratio[ii] = r_local

        one_plus_r = 1 + r_local
        r_clamped = max(r_local, -0.999)
        r_pow = r_clamped**const_budyko
        inside = 1 + r_pow
        root_term = inside ** (1 / const_budyko)
        fcomb = one_plus_r - root_term

        flat_factor[ii] = fcomb
        flat_out[ii] = fcomb * fw

    # Reshape to original shape
    fapar_max = flat_out.reshape(sza)

    return fapar_max


def plmodel_timeseries(
    A0_input: np.ndarray,
    fapar_max_input: np.ndarray,
    alpha_aet_pet: np.ndarray,
    time_step_days: int = 1,
    spinup_years: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Simulate seasonal LAI and fAPAR from potential GPP and aridity."""

    if A0_input.ndim != 3:
        raise ValueError("A0_input must be a 3-D array (row, col, time).")
    if fapar_max_input.ndim not in (2, 3):
        raise ValueError("fapar_max_input must be 2-D or 3-D.")
    if alpha_aet_pet.ndim != 2:
        raise ValueError("alpha_aet_pet must be a 2-D array (row, col).")

    A0_data = np.asarray(A0_input, dtype=np.float64)
    fapar_max = np.asarray(fapar_max_input, dtype=np.float64)
    alpha = np.asarray(alpha_aet_pet, dtype=np.float64)

    nrow, ncol, ntime = A0_data.shape
    if fapar_max.shape[0] != nrow or fapar_max.shape[1] != ncol:
        raise ValueError("Spatial dimensions of fapar_max_input must match A0_input.")
    if alpha.shape != (nrow, ncol):
        raise ValueError("Spatial dimensions of alpha_aet_pet must match A0_input.")

    const_cli_prc = 95
    const_min_a0 = 1.0e-6
    const_lagcoef = 64.6
    const_lagmax = 365

    if fapar_max.ndim == 3:
        fapar_max_grid = _nanmax_cube(fapar_max)
    else:
        fapar_max_grid = fapar_max.copy()
    np.clip(fapar_max_grid, 0.0, 0.999, out=fapar_max_grid)

    lai_max_grid = _cal_lai_from_fapar(fapar_max_grid)

    A0_data_prc_max = _nanpercentile(A0_data, const_cli_prc)
    A0_data_prc_max = np.where(
        A0_data_prc_max < const_min_a0, const_min_a0, A0_data_prc_max
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mratio_cal = lai_max_grid / A0_data_prc_max
    mratio_cal[~np.isfinite(mratio_cal)] = np.nan

    SLAI15DE1 = A0_data * mratio_cal[:, :, np.newaxis]

    lagday_grid = alpha * const_lagcoef
    lagday_grid[~np.isfinite(lagday_grid)] = 0.0
    lagday_grid = np.clip(lagday_grid, 0.0, const_lagmax)

    max_steps = int(round(const_lagmax / max(time_step_days, 1)))  # noqa: RUF046
    lag_steps_grid = np.rint(lagday_grid / max(time_step_days, 1)).astype(int)
    lag_steps_grid = np.clip(lag_steps_grid, 0, max_steps)

    num_spin = max(int(spinup_years), 1)
    SLAI_spin = np.tile(SLAI15DE1, (1, 1, num_spin))

    TLAI_spin = _apply_lag(SLAI_spin, lag_steps_grid)
    TLAI15DE1 = TLAI_spin[:, :, -ntime:]
    TLAI15DE1[TLAI15DE1 < 0] = 0.0

    fapar_sim = _cal_fapar_from_lai(TLAI15DE1)

    weighted_num, weighted_den = _weighted_sum(fapar_sim, A0_data)
    with np.errstate(divide="ignore", invalid="ignore"):
        fapar_weighted = weighted_num / weighted_den
    fapar_weighted[(~np.isfinite(weighted_den)) | (weighted_den <= 0)] = np.nan

    output_data = {
        "mratio_cal": mratio_cal,
        "SLAI15DE1": SLAI15DE1,
        "lagday_grid": lagday_grid,
        "lag_steps_grid": lag_steps_grid,
        "fapar_weighted": fapar_weighted,
        "time_step_days": np.array(time_step_days, dtype=np.int32),
        "spinup_years": np.array(num_spin, dtype=np.int32),
    }

    return TLAI15DE1, fapar_sim, output_data


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _nanmax_cube(data_cube: np.ndarray) -> np.ndarray:
    """Return the nan-aware maximum along the last axis."""
    max_vals = np.nanmax(data_cube, axis=2)
    mask_all_nan = ~np.any(np.isfinite(data_cube), axis=2)
    max_vals[mask_all_nan] = np.nan
    return max_vals


def _nanpercentile(data_cube: np.ndarray, percentile: float) -> np.ndarray:
    """Compute a nan-aware percentile along the last axis."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must lie between 0 and 100.")
    prc = np.nanpercentile(data_cube, percentile, axis=2)
    mask_all_nan = ~np.any(np.isfinite(data_cube), axis=2)
    prc[mask_all_nan] = np.nan
    return prc


def _apply_lag(data_cube: np.ndarray, lag_steps_grid: np.ndarray) -> np.ndarray:
    """Apply a running mean lag to each pixel individually."""
    nrow, ncol, ntime = data_cube.shape
    result = np.full((nrow, ncol, ntime), np.nan, dtype=np.float64)

    for row in range(nrow):
        for col in range(ncol):
            series = data_cube[row, col, :]
            if np.all(np.isnan(series)):
                continue
            lag_steps = int(lag_steps_grid[row, col])
            if lag_steps <= 0:
                result[row, col, :] = series
                continue

            values = series.copy()
            valid_mask = np.isfinite(values)
            values[~valid_mask] = 0.0
            counts = valid_mask.astype(np.float64)

            csum_values = np.cumsum(values)
            csum_counts = np.cumsum(counts)

            # DO tracking code start --------------------------
            do_totals_track = np.zeros_like(csum_values)
            do_counts_track = np.zeros_like(csum_values)
            do_idx_start_track = np.zeros_like(csum_values)

            # DO tracking code ends ---------------------------

            temp_result = np.empty(ntime, dtype=np.float64)
            for t in range(ntime):
                idx_start = t - lag_steps
                if idx_start < 0:
                    idx_start = 0
                total = csum_values[t] - (
                    csum_values[idx_start - 1] if idx_start > 0 else 0.0
                )
                count = csum_counts[t] - (
                    csum_counts[idx_start - 1] if idx_start > 0 else 0.0
                )
                if count > 0:
                    temp_result[t] = total / count
                else:
                    temp_result[t] = np.nan

                # DO tracking code start --------------------------
                do_totals_track[t] = total
                do_counts_track[t] = count
                do_idx_start_track[t] = idx_start
                # DO tracking code end ----------------------------

            result[row, col, :] = temp_result

    # At this point, the whole calculation can be saved out to file for a single site
    # using:
    #
    # import pandas as pd
    # df = pd.DataFrame(
    #     {
    #         "series": series.ravel(),
    #         "counts": counts.ravel(),
    #         "csum_values": csum_values.ravel(),
    #         "csum_counts": csum_counts.ravel(),
    #         "do_totals_track": do_totals_track.ravel(),
    #         "do_counts_track": do_counts_track.ravel(),
    #         "do_idx_start_track": do_idx_start_track.ravel(),
    #         "result": result.ravel(),
    #     }
    # )
    # df.to_csv("zhu_tracking.csv")

    return result


def _cal_lai_from_fapar(fapar: np.ndarray) -> np.ndarray:
    """Convert fAPAR to LAI using Beer-Lambert extinction (k=0.5)."""
    fapar_clipped = np.clip(fapar, 0.0, 0.999)
    return -2.0 * np.log(np.maximum(1.0 - fapar_clipped, 1.0e-6))


def _cal_fapar_from_lai(lai: np.ndarray) -> np.ndarray:
    """Convert LAI to fAPAR using Beer-Lambert extinction (k=0.5)."""
    return 1.0 - np.exp(-0.5 * np.clip(lai, 0.0, None))


def _weighted_sum(
    fapar_series: np.ndarray, A0_series: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return numerator and denominator of the A0-weighted mean fAPAR."""
    mask = np.isfinite(fapar_series) & np.isfinite(A0_series)
    weighted_num = np.nansum(np.where(mask, fapar_series * A0_series, 0.0), axis=2)
    weighted_den = np.nansum(np.where(np.isfinite(A0_series), A0_series, 0.0), axis=2)

    no_data_mask = ~np.any(np.isfinite(A0_series), axis=2)
    weighted_num[no_data_mask] = np.nan
    weighted_den[no_data_mask] = np.nan

    return weighted_num, weighted_den
