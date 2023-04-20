r"""The :mod:`~pyrealm.pmodel.subdaily` module provides extensions to the P Model that
incorporate modelling of the fast and slow responses of photosynthesis to changing
conditions.
"""  # noqa: D205, D415
from typing import Optional
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
)


def memory_effect(values: NDArray, alpha: float = 0.067) -> NDArray:
    r"""Apply a memory effect to a variable.

    Three key photosynthetic parameters (:math:`\xi`, :math:`V_{cmax25}` and
    :math:`J_{max25}`) show slow responses to changing environmental conditions and do
    not instantaneously adopt optimal values. This function applies a rolling weighted
    average to apply a lagged response to one of these parameters.

    The estimation uses the paramater `alpha` (:math:`\alpha`) to control the speed of
    convergence of the estimated values (:math:`E`) to the calculated optimal values
    (:math:`O`):

    .. math::

        E_{t} = E_{t-1}(1 - \alpha) + O_{t} \alpha

    For :math:`t_{0}`, the first value in the optimal values is used so :math:`E_{0} =
    O_{0}`.

    The ``values`` array can have multiple dimensions but the first dimension is always
    assumed to represent time and the memory effect is calculated only along the first
    dimension.

    Args:
        values: The values to apply the memory effect to.
        alpha: The relative weight applied to the most recent observation

    Returns:
        An array of the same shape as ``values`` with the memory effect applied.
    """

    # Initialise the output storage and set the first values to be a slice along the
    # first axis of the input values
    memory_values = np.empty_like(values, dtype=np.float32)
    memory_values[0] = values[0]

    # Loop over the first axis, in each case taking slices through the first axis of the
    # inputs. This handles arrays of any dimension.
    for idx in range(1, len(memory_values)):
        memory_values[idx] = memory_values[idx - 1] * (1 - alpha) + values[idx] * alpha

    return memory_values


class FastSlowPModel:
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
    :class:`~pyrealm.pmodel.pmodel.PModelEnvironment` instance must represent the time
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
      of acclimation.
    * The realised values are then filled back onto the original subdaily timescale,
      with :math:`V_{cmax}` and :math:`J_{max}` then being calculated from the slowly
      responding :math:`V_{cmax25}` and :math:`J_{max25}` and the actual subdaily
      temperature observations and :math:`c_i` calculated using realised values of
      :math:`\xi` but subdaily values in the other parameters.
    * Predictions of GPP are then made as in the standard P Model.

    Args:
        env: An instance of :class:`~pyrealm.pmodel.pmodel.PModelEnvironment`.
        fs_scaler: An instance of
          :class:`~pyrealm.pmodel.fast_slow_scaler.FastSlowScaler`.
        fapar: The :math:`f_{APAR}` for each observation.
        ppfd: The PPDF for each observation.
        alpha: The :math:`\alpha` weight.
        kphio: The quantum yield efficiency of photosynthesis (:math:`\phi_0`, -).
        fill_kind: The approach used to fill daily realised values to the subdaily
          timescale, currently one of 'previous' or 'linear'.
    """

    def __init__(
        self,
        env: PModelEnvironment,
        fs_scaler: FastSlowScaler,
        ppfd: NDArray,
        fapar: NDArray,
        alpha: float = 1 / 15,
        kphio: float = 1 / 8,
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

        self.env = env
        # Get the daily estimates of the acclimation targets for forcing variables
        temp_acclim = fs_scaler.get_daily_means(self.env.tc)
        co2_acclim = fs_scaler.get_daily_means(self.env.co2)
        patm_acclim = fs_scaler.get_daily_means(self.env.patm)
        vpd_acclim = fs_scaler.get_daily_means(self.env.vpd)

        # TODO - calculate the acclimated daily model using GPP per unit Iabs and then
        #        scale up to subdaily variation in fapar and ppfd at the end. This might
        #        then allow this implementation to move inside PModel with an optional
        #        fsscaler argument.
        self.ppfd_acclim = fs_scaler.get_daily_means(ppfd)
        self.fapar_acclim = fs_scaler.get_daily_means(fapar)

        # Calculate the PModelEnvironment for those conditions and then the PModel
        # itself to obtain estimates of jmax and vcmax
        pmodel_env_acclim = PModelEnvironment(
            tc=temp_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim,
            patm=patm_acclim,
            const=self.env.const,
        )
        self.pmodel_acclim: PModel = PModel(pmodel_env_acclim, kphio=kphio)
        r"""P Model predictions for the daily acclimation conditions.

        A :class:`~pyrealm.pmodel.pmodel.PModel` instance providing the predictions of
        the P Model for the daily acclimation conditions set for the FastSlowPModel. The
        model predicts instantaneous optimal estimates of :math:`V_{cmax}`,
        :math:`J_{max}` and :math:`\xi`, which are then used to estimate realised values
        of those parameters given slow responses to acclimation.
        """

        # Calculate productivity measures including jmax and vcmax
        self.pmodel_acclim.estimate_productivity(
            fapar=self.fapar_acclim, ppfd=self.ppfd_acclim
        )

        # Calculate the optimal jmax and vcmax at 25°C
        # TODO - Are these any of the existing values in the constants?
        ha_vcmax25 = 65330
        ha_jmax25 = 43900

        tk_acclim = temp_acclim + self.env.const.k_CtoK
        self.vcmax25_opt = self.pmodel_acclim.vcmax * (
            1 / calc_ftemp_arrh(tk_acclim, ha_vcmax25)
        )
        self.jmax25_opt = self.pmodel_acclim.jmax * (
            1 / calc_ftemp_arrh(tk_acclim, ha_jmax25)
        )

        # Calculate the realised values from the instantaneous optimal values
        self.xi_real: NDArray = memory_effect(self.pmodel_acclim.optchi.xi, alpha=alpha)
        r"""Realised daily slow responses in :math:`\xi`"""
        self.vcmax25_real: NDArray = memory_effect(self.vcmax25_opt, alpha=alpha)
        r"""Realised daily slow responses in :math:`V_{cmax25}`"""
        self.jmax25_real: NDArray = memory_effect(self.jmax25_opt, alpha=alpha)
        r"""Realised daily slow responses in :math:`J_{max25}`"""

        # Fill the daily realised values onto the subdaily scale
        subdaily_tk = self.env.tc + self.env.const.k_CtoK

        # Fill the realised xi, jmax25 and vcmax25 from subdaily to daily and then
        # adjust jmax25 and vcmax25 to jmax and vcmax given actual temperature at
        # subdaily timescale
        self.subdaily_vcmax25 = fs_scaler.fill_daily_to_subdaily(self.vcmax25_real)
        self.subdaily_jmax25 = fs_scaler.fill_daily_to_subdaily(self.jmax25_real)
        self.subdaily_xi = fs_scaler.fill_daily_to_subdaily(self.xi_real)

        self.subdaily_vcmax: NDArray = self.subdaily_vcmax25 * calc_ftemp_arrh(
            tk=subdaily_tk, ha=ha_vcmax25
        )
        """Estimated subdaily :math:`V_{cmax}`."""

        self.subdaily_jmax: NDArray = self.subdaily_jmax25 * calc_ftemp_arrh(
            tk=subdaily_tk, ha=ha_jmax25
        )
        """Estimated subdaily :math:`J_{max}`."""

        self.subdaily_ci: NDArray = (
            self.subdaily_xi * self.env.ca + self.env.gammastar * np.sqrt(self.env.vpd)
        ) / (self.subdaily_xi + np.sqrt(self.env.vpd))
        """Estimated subdaily :math:`c_i`."""

        # Calculate Ac, J and Aj at subdaily scale to calculate assimilation
        self.subdaily_Ac: NDArray = (
            self.subdaily_vcmax
            * (self.subdaily_ci - self.env.gammastar)
            / (self.subdaily_ci + self.env.kmm)
        )
        """Estimated subdaily :math:`A_c`."""

        kphio_tc = kphio * calc_ftemp_kphio(tc=self.env.tc)
        iabs = fapar * ppfd

        subdaily_J = (4 * kphio_tc * iabs) / np.sqrt(
            1 + ((4 * kphio_tc * iabs) / self.subdaily_jmax) ** 2
        )

        self.subdaily_Aj: NDArray = (
            (subdaily_J / 4)
            * (self.subdaily_ci - self.env.gammastar)
            / (self.subdaily_ci + 2 * self.env.gammastar)
        )
        """Estimated subdaily :math:`A_j`."""

        # Calculate GPP and convert from mol to gC
        self.gpp: NDArray = (
            np.minimum(self.subdaily_Aj, self.subdaily_Ac) * self.env.const.k_c_molmass
        )
        """Estimated subdaily GPP."""

    # def estimate_gpp(self, fapar: NDArray, ppfd: NDArray) -> None:
    #     """Estimate productivity"""
    #     iabs = fapar * ppfd

    #     self.gpp = self.lue * iabs


class FastSlowPModel_JAMES:
    r"""Fits the JAMES P Model incorporating fast and slow photosynthetic responses.

    This is alternative implementation of the P Model incorporating slow responses that
    duplicates the original implementation of the weighted-average approach of
    {cite:t}`mengoli:2022a`.

    The key difference is that :math:`\xi` does not have a slow response, with
    :math:`c_i` calculated using the daily optimal values during the acclimation window
    for :math:`\xi`, :math:`c_a` and :math:`\Gamma^{\ast}`  and subdaily variation in
    VPD. The main implementation in :class:`~pyrealm.pmodel.subdaily.FastSlowPModel`
    instead uses fast subdaily responses in :math:`c_a`, :math:`\Gamma^{\ast}` and VPD
    and realised slow responses in :math:`\xi`.

    In addition, the original implementation included some subtle differences. The extra
    arguments to this function allow those differences to be recreated:

    * The optimal daily acclimation values were calculated using a different window for
      VPD, using an exact noon value rather than the mean of the daily window. A
      separate scaler can be provided using ``vpd_scaler`` to implement this.
    * The daily fAPAR values are also not the same as the mean of the acclimation
      window, so these can be set independently using ``fapar_acclim``.
    * The subdaily values of :math:`J_{max25}` and :math:`V_{cmax25}` were not filled
      foward from the end of the acclimation window. The ``fill_from`` argument can be
      used to recreate this.

    Args:
        env: An instance of :class:`~pyrealm.pmodel.pmodel.PModelEnvironment`.
        fs_scaler: An instance of
          :class:`~pyrealm.pmodel.fast_slow_scaler.FastSlowScaler`.
        fapar: The :math:`f_{APAR}` for each observation.
        ppfd: The PPDF for each observation.
        alpha: The :math:`\alpha` weight.
        kphio: The quantum yield efficiency of photosynthesis (:math:`\phi_0`, -).
        vpd_scaler: An alternate
          :class:`~pyrealm.pmodel.fast_slow_scaler.FastSlowScaler` instance used to
          calculate daily acclimation conditions for VPD.
        fill_from: A :class:`numpy.timedelta64` object giving the time since midnight
          used for filling :math:`J_{max25}` and :math:`V_{cmax25}` to the subdaily
          timescale.
        fill_kind: The approach used to fill daily realised values to the subdaily
          timescale, currently one of 'previous' or 'linear'.
    """

    def __init__(
        self,
        env: PModelEnvironment,
        fs_scaler: FastSlowScaler,
        ppfd: NDArray,
        fapar: NDArray,
        alpha: float = 1 / 15,
        kphio: float = 1 / 8,
        vpd_scaler: Optional[FastSlowScaler] = None,
        fill_from: Optional[np.timedelta64] = None,
        fill_kind: str = "previous",
    ) -> None:
        # Really warn about the API
        warn(
            "FastSlowPModel_JAMES is for validation against an older implementation "
            "and is not for production use.",
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

        self.env = env
        # Get the daily estimates of the acclimation targets for forcing variables
        temp_acclim = fs_scaler.get_daily_means(self.env.tc)
        co2_acclim = fs_scaler.get_daily_means(self.env.co2)
        patm_acclim = fs_scaler.get_daily_means(self.env.patm)

        if vpd_scaler is not None:
            vpd_acclim = vpd_scaler.get_daily_means(self.env.vpd)
        else:
            vpd_acclim = fs_scaler.get_daily_means(self.env.vpd)

        # TODO - calculate the acclimated daily model using GPP per unit Iabs and then
        #        scale up to subdaily variation in fapar and ppfd at the endrun
        self.ppfd_acclim = fs_scaler.get_daily_means(ppfd)
        self.fapar_acclim = fs_scaler.get_daily_means(fapar)

        # Calculate the PModelEnvironment for those conditions and then the PModel
        # itself to obtain estimates of jmax and vcmax
        pmodel_env_acclim = PModelEnvironment(
            tc=temp_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim,
            patm=patm_acclim,
            const=self.env.const,
        )
        self.pmodel_acclim: PModel = PModel(pmodel_env_acclim, kphio=kphio)
        r"""P Model predictions for the daily acclimation conditions.

        A :class:`~pyrealm.pmodel.pmodel.PModel` instance providing the predictions of
        the P Model for the daily acclimation conditions set for the FastSlowPModel. The
        model predicts instantaneous optimal estimates of :math:`V_{cmax}`,
        :math:`J_max` and `:math:`\xi`, which are then used to estimate realised values
        of those parameters given slow responses to acclimation.
        """

        # Calculate productivity measures including jmax and vcmax
        self.pmodel_acclim.estimate_productivity(
            fapar=self.fapar_acclim, ppfd=self.ppfd_acclim
        )

        # Calculate the optimal jmax and vcmax at 25°C
        # TODO - Are these any of the existing values in the constants?
        ha_vcmax25 = 65330
        ha_jmax25 = 43900

        tk_acclim = temp_acclim + self.env.const.k_CtoK
        self.vcmax25_opt = self.pmodel_acclim.vcmax * (
            1 / calc_ftemp_arrh(tk_acclim, ha_vcmax25)
        )
        self.jmax25_opt = self.pmodel_acclim.jmax * (
            1 / calc_ftemp_arrh(tk_acclim, ha_jmax25)
        )

        # Calculate the realised values from the instantaneous optimal values
        self.vcmax25_real: NDArray = memory_effect(self.vcmax25_opt, alpha=alpha)
        r"""Realised daily slow responses in :math:`V_{cmax25}`"""
        self.jmax25_real: NDArray = memory_effect(self.jmax25_opt, alpha=alpha)
        r"""Realised daily slow responses in :math:`J_{max25}`"""

        # Calculate the daily xi value, which does not have a slow reponse in this
        # implementation.
        # - Calculate subdaily time series for gammastar, xi and ca, filled forwards
        #   from midnight
        subdaily_gammastar = fs_scaler.fill_daily_to_subdaily(
            self.pmodel_acclim.env.gammastar, fill_from=np.timedelta64(0, "h")
        )
        subdaily_xi = fs_scaler.fill_daily_to_subdaily(
            self.pmodel_acclim.optchi.xi, fill_from=np.timedelta64(0, "h")
        )
        subdaily_ca = fs_scaler.fill_daily_to_subdaily(
            self.pmodel_acclim.env.ca, fill_from=np.timedelta64(0, "h")
        )

        # Calculate ci using the daily optimal acclimated values for xi, ca and
        # gammastar and the actual daily variation in VPD.
        self.subdaily_ci = (
            subdaily_xi * subdaily_ca + subdaily_gammastar * np.sqrt(self.env.vpd)
        ) / (subdaily_xi + np.sqrt(self.env.vpd))
        """Estimated subdaily :math:`c_i`."""

        # Fill the daily realised values onto the subdaily scale
        subdaily_tk = self.env.tc + self.env.const.k_CtoK

        # Fill the realised xi, jmax25 and vcmax25 from subdaily to daily and then
        # adjust jmax25 and vcmax25 to jmax and vcmax given actual temperature at
        # subdaily timescale
        self.subdaily_vcmax25 = fs_scaler.fill_daily_to_subdaily(
            self.vcmax25_real, fill_from=fill_from
        )
        self.subdaily_jmax25 = fs_scaler.fill_daily_to_subdaily(
            self.jmax25_real, fill_from=fill_from
        )

        self.subdaily_vcmax: NDArray = self.subdaily_vcmax25 * calc_ftemp_arrh(
            tk=subdaily_tk, ha=ha_vcmax25
        )
        """Estimated subdaily :math:`V_{cmax}`."""

        self.subdaily_jmax: NDArray = self.subdaily_jmax25 * calc_ftemp_arrh(
            tk=subdaily_tk, ha=ha_jmax25
        )
        """Estimated subdaily :math:`J_{max}`."""

        # Calculate Ac, J and Aj at subdaily scale to calculate assimilation
        self.subdaily_Ac: NDArray = (
            self.subdaily_vcmax
            * (self.subdaily_ci - self.env.gammastar)
            / (self.subdaily_ci + self.env.kmm)
        )
        """Estimated subdaily :math:`A_c`."""

        kphio_tc = kphio * calc_ftemp_kphio(tc=self.env.tc)
        iabs = fapar * ppfd

        subdaily_J = (4 * kphio_tc * iabs) / np.sqrt(
            1 + ((4 * kphio_tc * iabs) / self.subdaily_jmax) ** 2
        )

        self.subdaily_Aj: NDArray = (
            (subdaily_J / 4)
            * (self.subdaily_ci - self.env.gammastar)
            / (self.subdaily_ci + 2 * self.env.gammastar)
        )
        """Estimated subdaily :math:`A_j`."""

        # Calculate GPP, converting from mol m2 s1 to grams carbon m2 s1
        self.gpp: NDArray = (
            np.minimum(self.subdaily_Aj, self.subdaily_Ac) * self.env.const.k_c_molmass
        )
        """Estimated subdaily GPP."""
