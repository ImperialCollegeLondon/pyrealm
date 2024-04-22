"""Runs a profiler on the subdaily pmodel to identify runtime bottlenecks."""

import numpy as np
import pytest


@pytest.mark.profiling
def test_profiling_subdaily(pmodel_profile_data):
    """Profiling the subdaily submodule."""
    from pyrealm.pmodel import SubdailyPModel, SubdailyScaler

    # Unpack feature components
    pm_env, fapar, ppfd, local_time = pmodel_profile_data

    # SubdailyPModel with 1 hour noon acclimation window
    fsscaler = SubdailyScaler(local_time)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(1, "h"),
    )
    subdaily_pmod = SubdailyPModel(
        env=pm_env,
        fs_scaler=fsscaler,
        allow_holdover=True,
        fapar=fapar,
        ppfd=ppfd,
        alpha=1 / 15,
    )
    return subdaily_pmod
