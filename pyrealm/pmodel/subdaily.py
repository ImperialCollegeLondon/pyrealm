r"""The :mod:`~pyrealm.pmodel.subdaily` module provides extensions to the P Model that
incorporate modelling of the fast and slow responses of photosynthesis to changing
conditions.

The initial implementation of the subdaily model followed the code structure of the
original implementation of the weighted mean approach :cite:p:`mengoli:2022a`, which was
hard-coded to applying the {cite:t}`Prentice:2014bc` equations for the C3 pathway and
contained some other slight differences in calculations. That original structure is
largely preserved in the :class:`~pyrealm.pmodel.subdaily.SubdailyPModel_JAMES` class,
which also documents the main differences and updates to the original approach. The
:class:`~pyrealm.pmodel.subdaily.SubdailyPModel_JAMES` implementation is not intended
for wide use and provides an approach for regression testing against outputs from the
original R implementation.

The main :class:`~pyrealm.pmodel.subdaily.SubdailyPModel` class provides the main
interface for fitting subdaily models. It incorporates slow responses of the :math:`\xi`
parameter in calculating optimal :math:`\chi`. This implementation can also be fitted
using any of the existing approaches for calculating optimal :math:`\chi`, including
using C4 pathways and soil moisture effects on optimal :math:`\chi`, but this is
experimental and not validated.
"""  # noqa: D205

from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm import ExperimentalFeatureWarning
from pyrealm.pmodel import (
    PModel,
    PModelEnvironment,
    SubdailyScaler,
    calc_ftemp_arrh,
    calc_ftemp_kphio,
)
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY


def memory_effect(
    values: NDArray, alpha: float = 0.067, allow_holdover: bool = False
) -> NDArray:
    r"""Apply a memory effect to a variable.

    Three key photosynthetic parameters (:math:`\xi`, :math:`V_{cmax25}` and
    :math:`J_{max25}`) show slow responses to changing environmental conditions and do
    not instantaneously adopt optimal values. This function applies a rolling weighted
    average to apply a lagged response to one of these parameters.

    The estimation uses the parameter `alpha` (:math:`\alpha`) to control the speed of
    convergence of the realised values (:math:`R`) to the calculated optimal values
    (:math:`O`):

    .. math::

        R_{t} = R_{t-1}(1 - \alpha) + O_{t} \alpha

    For :math:`t_{0}`, the first value in the optimal values is used so :math:`R_{0} =
    O_{0}`.

    The ``values`` array can have multiple dimensions but the first dimension is always
    assumed to represent time and the memory effect is calculated only along the first
    dimension.

    By default, the ``values`` array must not contain missing values (`numpy.nan`).
    However, :math:`V_{cmax}` and :math:`J_{max}` are not estimable in some conditions
    (namely when :math:`m \le c^{\ast}`, see
    :class:`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14`) and so missing values in
    P Model
    predictions can arise even when the forcing data is complete, breaking the recursion
    shown above. When ``allow_holdover=True``, initial missing values are kept, and the
    first observed optimal value is accepted as the first realised value (as with the
    start of the recursion above). After this, if the current optimal value is missing,
    then the previous estimate of the realised value is held over until it can next be
    updated from observed data.

    +-------------------+--------+-------------------------------------------------+
    |                   |        |   Current optimal (:math:`O_{t}`)               |
    +-------------------+--------+-----------------+-------------------------------+
    |                   |        |     NA          |   not NA                      |
    +-------------------+--------+-----------------+-------------------------------+
    | Previous          |    NA  |     NA          |     O_{t}                     |
    | realised          +--------+-----------------+-------------------------------+
    | (:math:`R_{t-1}`) | not NA | :math:`R_{t-1}` | :math:`R_{t-1}(1-a) + O_{t}a` |
    +-------------------+--------+-----------------+-------------------------------+

    Args:
        values: The values to apply the memory effect to.
        alpha: The relative weight applied to the most recent observation.
        allow_holdover: Allow missing values to be filled by holding over earlier
            values.

    Returns:
        An array of the same shape as ``values`` with the memory effect applied.
    """

    # Check for nan and nan handling
    nan_present = np.any(np.isnan(values))
    if nan_present and not allow_holdover:
        raise ValueError("Missing values in data passed to memory_effect")

    # Initialise the output storage and set the first values to be a slice along the
    # first axis of the input values
    memory_values = np.empty_like(values, dtype=np.float32)
    memory_values[0] = values[0]

    # Handle the data if there are no missing data,
    if not nan_present:
        # Loop over the first axis, in each case taking slices through the first axis of
        # the inputs. This handles arrays of any dimension.
        for idx in range(1, len(memory_values)):
            memory_values[idx] = (
                memory_values[idx - 1] * (1 - alpha) + values[idx] * alpha
            )

        return memory_values

    # Otherwise, do the same thing but handling missing data at each step.
    for idx in range(1, len(memory_values)):
        # Need to check for nan conditions:
        # - the previous value might be nan from an initial nan or sequence of nans, in
        #   which case the current value is accepted without weighting - it could be nan
        #   itself to extend a chain of initial nan values.
        # - the current value might be nan, in which case the previous value gets
        #   held over as the current value.
        prev_nan = np.isnan(memory_values[idx - 1])
        curr_nan = np.isnan(values[idx])
        memory_values[idx] = np.where(
            prev_nan,
            values[idx],
            np.where(
                curr_nan,
                memory_values[idx - 1],
                memory_values[idx - 1] * (1 - alpha) + values[idx] * alpha,
            ),
        )

    return memory_values


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

    The workflow of the model:

    * The first dimension of the data arrays used to create the
      :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance must
      represent the time axis of the observations. The ``fs_scaler`` argument is used to
      provide :class:`~pyrealm.pmodel.scaler.SubdailyScaler` instance which
      sets the dates and time of those observations and sets which daily observations
      form the daily acclimation window that will be used to estimate the optimal daily
      behaviour, using one of the ``set_`` methods to that class.
    * The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_daily_means` method
      is then used to extract daily average values for forcing variables from within the
      acclimation window, setting the conditions that the plant will optimise to.
    * A standard P Model is then run on those daily forcing values to generate predicted
      states for photosynthetic parameters that give rise to optimal productivity in
      that window.
    * The :meth:`~pyrealm.pmodel.subdaily.memory_effect` function is then used to
      calculate realised slowly responding values for :math:`\xi`, :math:`V_{cmax25}`
      and :math:`J_{max25}`, given a weight :math:`\alpha \in [0,1]` that sets the speed
      of acclimation using :math:`R_{t} = R_{t-1}(1 - \alpha) + O_{t} \alpha`, where
      :math:`O` is the optimal value and :math:`R` is the realised value after
      acclimation along a time series (:math:`t = 1..n`). Higher values of `alpha` give
      more rapid acclimation: :math:`\alpha=1` results in immediate acclimation and
      :math:`\alpha=0` results in no acclimation at all, with values pinned to the
      initial estimates.
    * The realised values are then filled back onto the original subdaily timescale,
      with :math:`V_{cmax}` and :math:`J_{max}` then being calculated from the slowly
      responding :math:`V_{cmax25}` and :math:`J_{max25}` and the actual subdaily
      temperature observations and :math:`c_i` calculated using realised values of
      :math:`\xi` but subdaily values in the other parameters.
    * Predictions of GPP are then made as in the standard P Model.

    Missing values:

        Missing data can arise in a number of ways: actual gaps in the forcing data, the
        observations starting part way through a day and missing some or all of the
        acclimation window for the day, or undefined values in P Model predictions. Some
        options include:

        * The ``allow_partial_data`` argument is passed on to
          :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_daily_means` to
          allow daily optimum conditions to be calculated when the data in the
          acclimation window is incomplete. This does not fix problems when no data is
          present in the window or when the P Model predictions for a day are undefined.

        * The ``allow_holdover`` argument is passed on to
          :meth:`~pyrealm.pmodel.subdaily.memory_effect` to set whether missing values
          in the optimal predictions can be filled by holding over previous valid
          values.

    Args:
        env: An instance of
          :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`
        fs_scaler: An instance of
          :class:`~pyrealm.pmodel.scaler.SubdailyScaler`.
        fapar: The :math:`f_{APAR}` for each observation.
        ppfd: The PPDF for each observation.
        alpha: The :math:`\alpha` weight.
        allow_holdover: Should the :func:`~pyrealm.pmodel.subdaily.memory_effect`
          function be allowed to hold over values to fill missing values.
        allow_partial_data: Should estimates of daily optimal conditions be calculated
          with missing values in the acclimation window.
        kphio: The quantum yield efficiency of photosynthesis (:math:`\phi_0`, -).
        fill_kind: The approach used to fill daily realised values to the subdaily
          timescale, currently one of 'previous' or 'linear'.
    """

    def __init__(
        self,
        env: PModelEnvironment,
        fs_scaler: SubdailyScaler,
        fapar: NDArray,
        ppfd: NDArray,
        kphio: float = 1 / 8,
        do_ftemp_kphio: bool = True,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        alpha: float = 1 / 15,
        allow_holdover: bool = False,
        allow_partial_data: bool = False,
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

        # Store the datetimes for reference
        self.datetimes = fs_scaler.datetimes
        """The datetimes of the observations used in the subdaily model."""

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
            *optimal_chi_class.requires,
        ]
        daily_environment: dict[str, NDArray] = {}
        for env_var_name in daily_environment_vars:
            env_var = getattr(self.env, env_var_name)
            if env_var is not None:
                daily_environment[env_var_name] = fs_scaler.get_daily_means(
                    env_var, allow_partial_data=allow_partial_data
                )

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
        the P Model for the daily acclimation conditions set for the SubdailyPModel. The
        model is used to obtain predictions of the instantaneously optimal estimates of
        :math:`V_{cmax}`, :math:`J_{max}` and :math:`\xi` during the acclimation
        window. These are then used to estimate realised values of those parameters
        given slow responses to acclimation.
        """

        # 3) Estimate productivity to calculate jmax and vcmax
        self.ppfd_acclim = fs_scaler.get_daily_means(
            ppfd, allow_partial_data=allow_partial_data
        )
        self.fapar_acclim = fs_scaler.get_daily_means(
            fapar, allow_partial_data=allow_partial_data
        )

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
            self.pmodel_acclim.optchi.xi, alpha=alpha, allow_holdover=allow_holdover
        )
        r"""Realised daily slow responses in :math:`\xi`"""
        self.vcmax25_real: NDArray = memory_effect(
            self.vcmax25_opt, alpha=alpha, allow_holdover=allow_holdover
        )
        r"""Realised daily slow responses in :math:`V_{cmax25}`"""
        self.jmax25_real: NDArray = memory_effect(
            self.jmax25_opt, alpha=alpha, allow_holdover=allow_holdover
        )
        r"""Realised daily slow responses in :math:`J_{max25}`"""

        # 6) Fill the realised xi, jmax25 and vcmax25 from daily values back to the
        # subdaily timescale.
        self.subdaily_vcmax25 = fs_scaler.fill_daily_to_subdaily(self.vcmax25_real)
        self.subdaily_jmax25 = fs_scaler.fill_daily_to_subdaily(self.jmax25_real)
        self.subdaily_xi = fs_scaler.fill_daily_to_subdaily(self.xi_real)

        # 7) Adjust subdaily jmax25 and vcmax25 back to jmax and vcmax given the
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


def convert_pmodel_to_subdaily(
    pmodel: PModel,
    fs_scaler: SubdailyScaler,
    alpha: float = 1 / 15,
    allow_holdover: bool = False,
    fill_kind: str = "previous",
) -> SubdailyPModel:
    r"""Convert a standard P Model to a subdaily P Model.

    This function takes an existing :class:`~pyrealm.pmodel.pmodel.PModel` instance and
    converts it to a :class:`~pyrealm.pmodel.subdaily.SubdailyPModel` instance
    using provided settings. The
    :meth:`~pyrealm.pmodel.pmodel.PModel.estimate_productivity` method must have been
    called on the :class:`~pyrealm.pmodel.pmodel.PModel` instance in order to provide
    PPFD and FAPAR to the subdaily model

    Args:
        pmodel: An existing standard PModel instance.
        fs_scaler: A SubdailyScaler instance giving the acclimation window for the
            subdaily model.
        alpha: The :math:`\alpha` weight.
        allow_holdover: Should the :func:`~pyrealm.pmodel.subdaily.memory_effect`
          function be allowed to hold over values to fill missing values.
        fill_kind: The approach used to fill daily realised values to the subdaily
          timescale, currently one of 'previous' or 'linear'.
    """
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
        allow_holdover=allow_holdover,
        fill_kind=fill_kind,
    )


class SubdailyPModel_JAMES:
    r"""Fits the JAMES P Model incorporating fast and slow photosynthetic responses.

    This is alternative implementation of the P Model incorporating slow responses that
    duplicates the original implementation of the weighted-average approach of
    {cite:t}`mengoli:2022a`.

    The key difference is that :math:`\xi` does not have a slow response, with
    :math:`c_i` calculated using the daily optimal values during the acclimation window
    for :math:`\xi`, :math:`c_a` and :math:`\Gamma^{\ast}`  and subdaily variation in
    VPD. The main implementation in :class:`~pyrealm.pmodel.subdaily.SubdailyPModel`
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
        env: An instance of
          :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`
        fs_scaler: An instance of
          :class:`~pyrealm.pmodel.scaler.SubdailyScaler`.
        fapar: The :math:`f_{APAR}` for each observation.
        ppfd: The PPDF for each observation.
        alpha: The :math:`\alpha` weight.
        allow_holdover: Should the :func:`~pyrealm.pmodel.subdaily.memory_effect`
          function be allowed to hold over values to fill missing values.
        kphio: The quantum yield efficiency of photosynthesis (:math:`\phi_0`, -).
        vpd_scaler: An alternate
          :class:`~pyrealm.pmodel.scaler.SubdailyScaler` instance used to
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
        fs_scaler: SubdailyScaler,
        ppfd: NDArray,
        fapar: NDArray,
        alpha: float = 1 / 15,
        allow_holdover: bool = False,
        kphio: float = 1 / 8,
        vpd_scaler: SubdailyScaler | None = None,
        fill_from: np.timedelta64 | None = None,
        fill_kind: str = "previous",
    ) -> None:
        # Really warn about the API
        warn(
            "SubdailyPModel_JAMES is for validation against an older implementation "
            "and is not for production use.",
            DeprecationWarning,
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
            pmodel_const=self.env.pmodel_const,
            core_const=self.env.core_const,
        )
        self.pmodel_acclim: PModel = PModel(pmodel_env_acclim, kphio=kphio)
        r"""P Model predictions for the daily acclimation conditions.

        A :class:`~pyrealm.pmodel.pmodel.PModel` instance providing the predictions of
        the P Model for the daily acclimation conditions set for the SubdailyPModel. The
        model predicts instantaneous optimal estimates of :math:`V_{cmax}`,
        :math:`J_max` and `:math:`\xi`, which are then used to estimate realised values
        of those parameters given slow responses to acclimation.
        """

        # Calculate productivity measures including jmax and vcmax
        self.pmodel_acclim.estimate_productivity(
            fapar=self.fapar_acclim, ppfd=self.ppfd_acclim
        )

        # Calculate the optimal jmax and vcmax at 25°C
        tk_acclim = temp_acclim + self.env.core_const.k_CtoK
        self.vcmax25_opt = self.pmodel_acclim.vcmax * (
            1 / calc_ftemp_arrh(tk_acclim, self.env.pmodel_const.subdaily_vcmax25_ha)
        )
        self.jmax25_opt = self.pmodel_acclim.jmax * (
            1 / calc_ftemp_arrh(tk_acclim, self.env.pmodel_const.subdaily_jmax25_ha)
        )

        # Calculate the realised values from the instantaneous optimal values
        self.vcmax25_real: NDArray = memory_effect(
            self.vcmax25_opt, alpha=alpha, allow_holdover=allow_holdover
        )
        r"""Realised daily slow responses in :math:`V_{cmax25}`"""
        self.jmax25_real: NDArray = memory_effect(
            self.jmax25_opt, alpha=alpha, allow_holdover=allow_holdover
        )
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
        subdaily_tk = self.env.tc + self.env.core_const.k_CtoK

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
            tk=subdaily_tk, ha=self.env.pmodel_const.subdaily_vcmax25_ha
        )
        """Estimated subdaily :math:`V_{cmax}`."""

        self.subdaily_jmax: NDArray = self.subdaily_jmax25 * calc_ftemp_arrh(
            tk=subdaily_tk, ha=self.env.pmodel_const.subdaily_jmax25_ha
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
            np.minimum(self.subdaily_Aj, self.subdaily_Ac)
            * self.env.core_const.k_c_molmass
        )
        """Estimated subdaily GPP."""
