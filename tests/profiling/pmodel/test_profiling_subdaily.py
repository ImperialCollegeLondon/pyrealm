"""Runs a profiler on the subdaily pmodel to identify runtime bottlenecks."""

import numpy as np
import pytest


@pytest.mark.profiling
def test_profiling_subdaily(pmodel_profile_data):
    """Profiling the subdaily submodule."""
    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.new_pmodel import SubdailyPModelNew

    # Unpack feature components
    pm_env, local_time = pmodel_profile_data

    # SubdailyPModel with 1 hour noon acclimation window
    acclim_model = AcclimationModel(local_time, allow_holdover=True, alpha=1 / 15)
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"), half_width=np.timedelta64(1, "h")
    )
    subdaily_pmod = SubdailyPModelNew(
        env=pm_env,
        acclim_model=acclim_model,
    )
    return subdaily_pmod
