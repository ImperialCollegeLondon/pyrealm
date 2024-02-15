r"""The :mod:`~pyrealm.pmodel.subdaily` module provides extensions to the P Model that
incorporate modelling of the fast and slow responses of photosynthesis to changing
conditions.
"""  # noqa: D205, D415

from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm import ExperimentalFeatureWarning
from pyrealm.pmodel import (
    FastSlowScaler,
    PModel,
    PModelEnvironment,
    calc_ftemp_arrh,
    calc_ftemp_kphio,
    memory_effect,
)
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY


class SubdailyPModel:
    r"""Fit a P Model incorporating fast and slow photosynthetic responses.

    The :class:`~pyrealm.pmodel.pmodel.PModel` implementation of the P Model assumes
    that plants instantaneously adopt optimal behaviour, which is reasonable where the
    data represents average conditions over longer timescales and the plants can be
    assumed to have acclimated to optimal behaviour. Over shorter timescales, this
    assumption is unwarranted and photosynthetic slow responses need to be included.
    This class implements the weighted-average approach of {cite:t}`mengoli:2022a`, but
    is extended to include the slow response of :math:`\xi` in addition to
    :math:`V_{cmax25}` and :math:`J_{max25}`.

    The first dimension of the data arrays use to create the
    :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance must
    represent
    the time
    axis of the observations. The actual datetimes of those observations must then be
    used to initialiase a :class:`~pyrealm.pmodel.fast_slow_scaler.FastSlowScaler`
    instance, and one of the ``set_`` methods of that class must be used to define an
    acclimation window.

    The workflow of the model is then:

    * The daily acclimation window set in the
      :class:`~pyrealm.pmodel.fast_slow_scaler.FastSlowScaler` instance is used to
      calculate daily average values for forcing variables from within the acclimation
      window.
    * A standard P Model is then run on those daily forcing values to generate predicted
      states for photosynthetic parameters that give rise to optimal productivity.
    * The :meth:`~pyrealm.pmodel.subdaily.memory_effect` function is then used to
      calculate realised slowly responding values for :math:`\xi`, :math:`V_{cmax25}`
      and :math:`J_{max25}`, given a weight :math:`\alpha \in [0,1]` that sets the speed
      of acclimation. The ``handle_nan`` argument is passed to this function to set
      whether missing values in the optimal predictions are permitted and handled.
    * The realised values are then filled back onto the original subdaily timescale,
      with :math:`V_{cmax}` and :math:`J_{max}` then being calculated from the slowly
      responding :math:`V_{cmax25}` and :math:`J_{max25}` and the actual subdaily
      temperature observations and :math:`c_i` calculated using realised values of
      :math:`\xi` but subdaily values in the other parameters.
    * Predictions of GPP are then made as in the standard P Model.

    Args:
        env: An instance of
          :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`
        fs_scaler: An instance of
          :class:`~pyrealm.pmodel.fast_slow_scaler.FastSlowScaler`.
        fapar: The :math:`f_{APAR}` for each observation.
        ppfd: The PPDF for each observation.
        alpha: The :math:`\alpha` weight.
        handle_nan: Should the :func:`~pyrealm.pmodel.subdaily.memory_effect` function
          be allowe to handle missing values.
        kphio: The quantum yield efficiency of photosynthesis (:math:`\phi_0`, -).
        fill_kind: The approach used to fill daily realised values to the subdaily
          timescale, currently one of 'previous' or 'linear'.
    """

    def __init__(
        self,
        env: PModelEnvironment,
        fs_scaler: FastSlowScaler,
        fapar: NDArray,
        ppfd: NDArray,
        kphio: float = 1 / 8,
        do_ftemp_kphio: bool = True,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        alpha: float = 1 / 15,
        handle_nan: bool = False,
        fill_kind: str = "previous",
    ) -> None:
        # Warn about the API
        warn(
            "This is a draft implementation and the API and calculations may change",
            ExperimentalFeatureWarning,
        )

        # Check that the length of the fast slow scaler is congruent with the
        # first axis of the photosynthetic environment
        n_datetimes = fs_scaler.datetimes.shape[0]
        n_env_first_axis = env.tc.shape[0]

        if n_datetimes != n_env_first_axis:
            raise ValueError("env and fs_scaler do not have congruent dimensions")

        # Has a set method been run on the fast slow scaler
        if not hasattr(fs_scaler, "include"):
            raise ValueError("The daily sampling window has not been set on fs_scaler")

        # Set up kphio attributes
        self.env: PModelEnvironment = env
        self.init_kphio: float = kphio
        self.do_ftemp_kphio = do_ftemp_kphio
        self.kphio: NDArray

        # 1) Generate a PModelEnvironment containing the average conditions within the
        #    daily acclimation window, including any optional variables required by the
        #    optimal chi calculations used in the model.
        optimal_chi_class = OPTIMAL_CHI_CLASS_REGISTRY[method_optchi]
        daily_environment_vars = [
            "tc",
            "co2",
            "patm",
            "vpd",
        ] + optimal_chi_class.requires
        daily_environment: dict[str, NDArray] = {}
        for env_var_name in daily_environment_vars:
            env_var = getattr(self.env, env_var_name)
            if env_var is not None:
                daily_environment[env_var_name] = fs_scaler.get_daily_means(env_var)

        pmodel_env_acclim = PModelEnvironment(
            **daily_environment,
            pmodel_const=self.env.pmodel_const,
            core_const=self.env.core_const,
        )

        # 2) Fit a PModel to those environmental conditions, using the supplied settings
        #    for the original model.
        self.pmodel_acclim: PModel = PModel(
            pmodel_env_acclim,
            kphio=kphio,
            do_ftemp_kphio=do_ftemp_kphio,
            method_optchi=method_optchi,
            method_jmaxlim=method_jmaxlim,
        )
        r"""P Model predictions for the daily acclimation conditions.

        A :class:`~pyrealm.pmodel.pmodel.PModel` instance providing the predictions of
        the P Model for the daily acclimation conditions set for the FastSlowPModel. The
        model is used to obtain predictions of the instantaneously optimal estimates of
        :math:`V_{cmax}`, :math:`J_{max}` and :math:`\xi` during the acclimation window.
        These are then used to estimate realised values of those parameters given slow
        responses to acclimation.
        """

        # 3) Estimate productivity to calculate jmax and vcmax
        self.ppfd_acclim = fs_scaler.get_daily_means(ppfd)
        self.fapar_acclim = fs_scaler.get_daily_means(fapar)

        self.pmodel_acclim.estimate_productivity(
            fapar=self.fapar_acclim, ppfd=self.ppfd_acclim
        )

        # 4) Calculate the optimal jmax and vcmax at 25°C
        tk_acclim = pmodel_env_acclim.tc + self.env.core_const.k_CtoK
        self.vcmax25_opt = self.pmodel_acclim.vcmax * (
            1 / calc_ftemp_arrh(tk_acclim, self.env.pmodel_const.subdaily_vcmax25_ha)
        )
        self.jmax25_opt = self.pmodel_acclim.jmax * (
            1 / calc_ftemp_arrh(tk_acclim, self.env.pmodel_const.subdaily_jmax25_ha)
        )

        # 5) Calculate the realised daily values from the instantaneous optimal values
        self.xi_real: NDArray = memory_effect(
            self.pmodel_acclim.optchi.xi, alpha=alpha, handle_nan=handle_nan
        )
        r"""Realised daily slow responses in :math:`\xi`"""
        self.vcmax25_real: NDArray = memory_effect(
            self.vcmax25_opt, alpha=alpha, handle_nan=handle_nan
        )
        r"""Realised daily slow responses in :math:`V_{cmax25}`"""
        self.jmax25_real: NDArray = memory_effect(
            self.jmax25_opt, alpha=alpha, handle_nan=handle_nan
        )
        r"""Realised daily slow responses in :math:`J_{max25}`"""

        # 6) Fill the realised xi, jmax25 and vcmax25 from daily values back to the
        # subdaily timescale.
        self.subdaily_vcmax25 = fs_scaler.fill_daily_to_subdaily(self.vcmax25_real)
        self.subdaily_jmax25 = fs_scaler.fill_daily_to_subdaily(self.jmax25_real)
        self.subdaily_xi = fs_scaler.fill_daily_to_subdaily(self.xi_real)

        # 7) Adjust subdaily jmax25 and vcmax25 back toto jmax and vcmax given the
        #    actual subdaily temperatures.
        subdaily_tk = self.env.tc + self.env.core_const.k_CtoK
        self.subdaily_vcmax: NDArray = self.subdaily_vcmax25 * calc_ftemp_arrh(
            tk=subdaily_tk, ha=self.env.pmodel_const.subdaily_vcmax25_ha
        )
        """Estimated subdaily :math:`V_{cmax}`."""

        self.subdaily_jmax: NDArray = self.subdaily_jmax25 * calc_ftemp_arrh(
            tk=subdaily_tk, ha=self.env.pmodel_const.subdaily_jmax25_ha
        )
        """Estimated subdaily :math:`J_{max}`."""

        # 8) Recalculate chi using the OptimalChi class from the provided method.
        self.optimal_chi = optimal_chi_class(
            env=self.env, pmodel_const=env.pmodel_const
        )
        self.optimal_chi.estimate_chi(xi_values=self.subdaily_xi)

        """Estimated subdaily :math:`c_i`."""

        # Calculate Ac, J and Aj at subdaily scale to calculate assimilation
        if self.do_ftemp_kphio:
            ftemp_kphio = calc_ftemp_kphio(
                env.tc, optimal_chi_class.is_c4, pmodel_const=env.pmodel_const
            )
            self.kphio = self.init_kphio * ftemp_kphio
        else:
            self.kphio = np.array([self.init_kphio])

        self.subdaily_Ac: NDArray = self.subdaily_vcmax * self.optimal_chi.mc
        """Estimated subdaily :math:`A_c`."""

        iabs = fapar * ppfd

        subdaily_J = (4 * self.kphio * iabs) / np.sqrt(
            1 + ((4 * self.kphio * iabs) / self.subdaily_jmax) ** 2
        )

        self.subdaily_Aj: NDArray = (subdaily_J / 4) * self.optimal_chi.mj
        """Estimated subdaily :math:`A_j`."""

        # Calculate GPP and convert from mol to gC
        self.gpp: NDArray = (
            np.minimum(self.subdaily_Aj, self.subdaily_Ac)
            * self.env.core_const.k_c_molmass
        )
        """Estimated subdaily GPP."""


def pmodel_to_subdaily(
    pmodel: PModel,
    fs_scaler: FastSlowScaler,
    alpha: float = 1 / 15,
    handle_nan: bool = False,
    fill_kind: str = "previous",
) -> SubdailyPModel:
    """Draft function to convert standard P Model to subdaily P Model."""
    # Check that productivity has been estimated

    return SubdailyPModel(
        env=pmodel.env,
        fs_scaler=fs_scaler,
        fapar=pmodel.fapar,
        ppfd=pmodel.ppfd,
        kphio=pmodel.init_kphio,
        do_ftemp_kphio=pmodel.do_ftemp_kphio,
        method_optchi=pmodel.method_optchi,
        method_jmaxlim=pmodel.method_jmaxlim,
        alpha=alpha,
        handle_nan=handle_nan,
        fill_kind=fill_kind,
    )
