"""Class to compute the fAPAR_max and annual peak Leaf Area Index (LAI)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import lambertw  # type: ignore[import-untyped]

from pyrealm.constants import PhenologyConst
from pyrealm.core.experimental import warn_experimental
from pyrealm.core.time_series import AnnualValueCalculator
from pyrealm.core.utilities import check_input_shapes, exponential_moving_average
from pyrealm.pmodel.pmodel import PModel, PModelABC, SubdailyPModel


class FaparLimitation:
    r"""Compute maximum annual fAPAR and LAI.

    This class calculates maximum annual fAPAR and LAI, following :cite:`cai:2025a`.
    The maximum annual fAPAR is calculated as the minimum of two terms capturing
    energy-limited and water limited fAPAR:

    .. math::

        \text{fAPAR}_{max} = \min{
                \left(1 - z / \left(k A_0 \right) \right),
                \left( c_a \left( 1 - \chi \right) / 1.6 D \right)
                \left( f_0 P / A_0 \right)
            }

    The maximum annual LAI is then calculated using Beer's law:

    .. math::

        \text{LAI}_{max} = - ( 1 / k ) \ln {1 - \text{fAPAR}_{max}}

    The class also calculates the parameter :math:`m`, which is the steady state annual
    ratio of leaf area index to GPP:

    .. math::

        m = \frac{ \sigma G \text{LAI}_{max}}{A_0 \text{fAPAR}_{max}}

    The :class:`~pyrealm.constants.phenology_const.PhenologyConst` class provides values
    for the following constants:

    * :math:`z` accounts for the growth and maintenance costs of leaves.
    * :math:`k` is the light extinction coefficient.
    * :math:`f_0` is is the ratio of annual total transpiration of annual total
      precipitation, calculated from the climatological aridity index (AI) (see
      :class:`PhenologyConst.calculate_f0<pyrealm.constants.phenology_const.PhenologyConst.calculate_f0>`).
    * :math:`\sigma` is a proportion that captures the departure of :math:`m` from the
      maximum due to biological delays in deploying and dropping the canopy during the
      growing season.

    The other variables are the required arguments to the class defined below. The most
    common source of these variables is from a P Model, and the
    :meth:`~pyrealm.phenology.fapar_limitation.FaparLimitation.from_pmodel` method can
    be used to create an instance directly from a fitted P Model.

    Args:
        annual_total_potential_gpp: The annual sum of potential GPP (:math:`A_0,
            \text{mol C m}^{-2} \text{year}^{-1}`)
        annual_mean_ca: The ambient CO2 partial pressure during the growing season
            (:math:`c_a`, Pa)
        annual_mean_chi: The annual mean ratio of ambient to leaf CO2 partial during the
            growing season (:math:`\chi`, Pa)
        annual_mean_vpd: The annual mean vapour pressure deficit during the growing
            season (:math:`D`, Pa)
        annual_total_precip: The annual total precipitation (:math:`P, \text{mol m}^{-2}
            \text{year}^{-1}`)
        annual_growing_season_length: The length of the growing season in days for each
            year (:math:`G`, days)
        aridity_index: A climatological estimate of the local aridity index, calculated
            as the long term (typically 20 years) total PET over total precipitation
            (:math:`AI`, unitless)
        phenology_const: An instance of
            :class:`~pyrealm.constants.phenology_const.PhenologyConst`
    """

    __experimental__ = True

    def __init__(
        self,
        annual_total_potential_gpp: NDArray[np.float64],
        annual_mean_ca: NDArray[np.float64],
        annual_mean_chi: NDArray[np.float64],
        annual_mean_vpd: NDArray[np.float64],
        annual_total_precip: NDArray[np.float64],
        annual_growing_season_length: NDArray[np.float64],
        years: NDArray[np.datetime64],
        aridity_index: NDArray[np.float64],
        phenology_const: PhenologyConst = PhenologyConst(),
    ) -> None:
        # Experimental class
        warn_experimental("FaparLimitation")

        # Validate the input shapes.
        check_input_shapes(
            annual_total_potential_gpp,
            annual_mean_ca,
            annual_mean_chi,
            annual_mean_vpd,
            annual_total_precip,
            aridity_index,
            annual_growing_season_length,
            years,
        )

        # Check the years values - must be datetime64[Y].
        # TODO - this is a bit stringent, but is more robust
        if not years.dtype == "<M8[Y]":
            raise ValueError("The years argument must provide np.datetime64[Y] values")

        self.years = years
        r"""The year of each observation."""
        self.annual_total_potential_gpp = annual_total_potential_gpp
        r"""The annual sum of potential GPP 
        (:math:`A_0, \text{mol C m}^{-2} \text{year}^{-1}`)"""
        self.annual_mean_ca = annual_mean_ca
        r"""Ambient CO2 partial pressure during the growing season (:math:`c_a`, Pa)"""
        self.annual_mean_chi = annual_mean_chi
        r"""Annual mean ratio of ambient to leaf CO2 partial during the 
        growing season (:math:`\chi`, Pa)"""
        self.annual_mean_vpd = annual_mean_vpd
        r"""Annual mean vapour pressure deficit during the growing season (:math:`D`,
        Pa)"""
        self.annual_total_precip = annual_total_precip
        r"""Annual total precipitation
        (:math:`P, \text{mol m}^{-2} \text{year}^{-1}`)"""
        self.annual_growing_season_length = annual_growing_season_length
        r"""Annual growing season length (:math:`G`, days)"""
        self.aridity_index = aridity_index
        r"""Climatological estimate of local aridity index (AI, unitless)"""

        # Make sure the aridity index is not zero
        if aridity_index <= 0:
            raise ValueError("The aridity index has to be positive.")

        # Constants used for phenology computations
        self.phenology_const = phenology_const

        #  f_0 is the ratio of annual total transpiration of annual total
        #  precipitation, which is an empirical function of the climatic Aridity Index
        #  (AI).
        f_0 = self.phenology_const.calculate_f0(aridity_index=self.aridity_index)

        # Calculate the energy and water limited terms.
        fapar_energylim = 1.0 - self.phenology_const.z / (
            self.phenology_const.k * annual_total_potential_gpp
        )
        fapar_waterlim = (
            f_0
            * annual_total_precip
            * annual_mean_ca
            * (1 - annual_mean_chi)
            / (1.6 * annual_mean_vpd * annual_total_potential_gpp)
        )

        self.fapar_max: NDArray[np.floating] = np.minimum(
            fapar_waterlim, fapar_energylim
        )
        """Estimated annual maximum fAPAR (unitless)."""
        self.energy_limited: NDArray[np.bool_] = fapar_energylim < fapar_waterlim
        """Boolean array showing if annual :math:`fAPAR_{max}` is water or energy
        limited."""
        self.annual_precip_molar: NDArray[np.floating] = annual_total_precip
        """The annual total precipitation for each year (moles year-1)."""

        self.lai_max: NDArray[np.floating] = -(1 / self.phenology_const.k) * np.log(
            1.0 - self.fapar_max
        )
        """Estimated annual maximum LAI (unitless)"""

        self.lai_to_gpp_ratio_m = (
            self.phenology_const.sigma
            * self.annual_growing_season_length
            * self.lai_max
        ) / (self.annual_total_potential_gpp * self.fapar_max)
        """The steady state ratio of leaf area index to potential GPP (:math:`m`)"""

    @classmethod
    def from_pmodel(
        cls,
        pmodel: PModelABC,
        growing_season: NDArray[np.bool],
        precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
        datetimes: NDArray[np.datetime64] | None = None,
        gpp_penalty_factor: NDArray[np.float64] | None = None,
        phenology_const: PhenologyConst = PhenologyConst(),
    ) -> FaparLimitation:
        r"""Create a FaparLimitation instance from a P Model and other inputs.

        The annual summary values of :math:`A_0, c_a, \chi` and :math:`D` used by the
        :meth:`~pyrealm.phenology.fapar_limitation.FaparLimitation` class can be taken
        directly from the predictions of a P Model. This method automatically extracts
        the required data from a fitted P Model and returns a ``FaparLimitation``
        instance.

        .. NOTE::

          The calculation of fAPAR limitation requires estimates of **potential** GPP,
          so the :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance
          used to fit the model **must** set ``fapar`` to be one.

        Some additional information is needed:

        * The calculation requires annual summaries of variables, so the ``datetimes``
          argument must be used to provide an array of datetimes for each observation.

        * The annual mean values :math:`c_a, \chi` and :math:`D` should be estimated
          during the growing season, so the ``growing_season`` argument must be used to
          provide a boolean value indicating which observations should be treated as in
          the growing season.

        * The calculation requires estimates of precipitation, so the ``precipitation``
          argument must provide estimates of total precipitation during each
          observations in moles of water per metre squared.

        * The calculation of the :math:`f_0` parameter requires estimates of site
          specific aridity index.

        The method accepts both standard and subdaily P Models and automatically uses
        the actual time intervals between observations to calculate the required
        weighted annual means and sums. This might lead to unexpected values: the yearly
        mean of monthly values :math:`1, 2, \dots, 12` would not be 6.5 because the
        monthly values are weighted according to the length of the month.

        Lastly, potential GPP is taken directly from the P Model instance. If you want
        to apply a post-hoc penalty factor to GPP (e.g. a water limitation factor), then
        you can optionally provide per-observation penalty estimates and they will be
        applied when calculating annual total potential assimilation.

        Args:
            pmodel: A :class:`pyrealm.pmodel.pmodel.PModel` or
                :class:`pyrealm.pmodel.pmodel.SubdailyPModel` instance, fitted with
                ``fapar`` fixed at one.
            datetimes: An array giving the datetimes of observations.
            growing_season: A boolean array indicating which observations are to be
                considered as part of the growing season.
            precip: An array of precipitation for each observation.
            aridity_index: A climatological estimate of local aridity index.
            gpp_penalty_factor: A post-hoc penalty factor to be applied to estimated
                GPP.
            phenology_const: An instance of
                :class:`~pyrealm.constants.phenology_const.PhenologyConst`
        """

        # Check the datetimes - should they be taken from the AcclimationModel of the
        # SubdailyPModel or are they required for standard PModels?
        if isinstance(pmodel, SubdailyPModel):
            if datetimes is not None:
                raise ValueError(
                    "Observation datetimes are not required with SubdailyPModel "
                    "inputs, the acclimation model datetimes are used."
                )
            datetimes = pmodel.acclim_model.datetimes

        elif isinstance(pmodel, PModel):
            if datetimes is None:
                raise ValueError(
                    "Observation datetimes are required with PModel inputs."
                )

        # Create the annual value calculator
        # - the code above guards against datetimes being None
        avc = AnnualValueCalculator(
            timing=datetimes,  # type: ignore [arg-type]
            growing_season=growing_season,
        )

        # Get the total GPP for each observation
        # - also need to handle missing values, easier to take _mean_ annual value
        #   and scale it up to an annual total
        # - TODO - handle incompleteness - when do we stop estimating annual values from
        #   partial years (or at least warn about it)

        # Extract GPP and apply any observation level penalty factor
        total_gpp = pmodel.gpp
        if gpp_penalty_factor is not None:
            total_gpp *= gpp_penalty_factor

        # Calculate annual mean potential GPP and scale up to the year
        annual_mean_potential_gpp = avc.get_annual_means(total_gpp)
        annual_total_potential_gpp = (
            annual_mean_potential_gpp * (avc.year_n_days) * 86400 * 1e-6
        ) / pmodel.core_const.k_c_molmass

        # Calculate annual mean ca, chi and VPD within growing season
        annual_mean_ca = avc.get_annual_means(pmodel.env.ca, within_growing_season=True)
        annual_mean_chi = avc.get_annual_means(
            pmodel.optchi.chi, within_growing_season=True
        )
        annual_mean_vpd = avc.get_annual_means(
            pmodel.env.vpd, within_growing_season=True
        )

        # Calculate total annual precipitation
        annual_total_precip = avc.get_annual_totals(precip)

        return cls(
            annual_total_potential_gpp=annual_total_potential_gpp,
            annual_mean_ca=annual_mean_ca,
            annual_mean_chi=annual_mean_chi,
            annual_mean_vpd=annual_mean_vpd,
            annual_total_precip=annual_total_precip,
            annual_growing_season_length=avc.year_n_growing_days,
            years=avc.years,
            aridity_index=aridity_index,
            phenology_const=phenology_const,
        )


class Phenology:
    """Phenology calculation."""

    def __init__(
        self,
        daily_gpp: NDArray[np.floating],
        datetimes: NDArray[np.datetime64],
        fapar_limitation: FaparLimitation,
        alpha: float = 1 / 15,
        phenology_const: PhenologyConst = PhenologyConst(),
    ):
        # Check the array input shapes
        check_input_shapes(daily_gpp, datetimes)

        # Check the datetimes provide ordered daily resolution observations - don't
        # insist on daily precision but check that second level representations of the
        # days is consistent with daily observations and strictly increases.
        datetimes = datetimes.astype("datetime64[s]")
        datetime_spacing = np.diff(datetimes)

        if not np.all(np.equal(datetime_spacing, 86400)):
            raise ValueError(
                "The datetimes argument must provide timestamps of a "
                "complete increasing daily time series"
            )

        # Get the years of observations and check they are all represnted in the
        # FaparLimitation object
        datetimes_year = datetimes.astype("datetime64[Y]")
        observation_years = np.unique(datetimes_year)

        missing_years = set(observation_years).difference(fapar_limitation.years)
        if missing_years:
            raise ValueError(
                f"The observation datetimes include years that are not included in the "
                f"fapar_limitation data: {', '.join([str(y) for y in missing_years])}"
            )

        # Store values
        self.alpha = alpha
        """The alpha value used with the exponential weighted average to set the
        magnitude of the lagging between realised LAI and the steady state values."""
        self.phenology_const = phenology_const
        """The phenology constants used in the model."""

        # Calculate the index of each observation in the FaparLimitation years.
        # This uses a shortcut to avoid looking up using np.where or np.searchsorted:
        # * The FaparLimitation.years are a sequence of values: N, N+1, N+2, ..., N+M
        # * The datetime_years are now validated to lie in N, ..., N+M
        # * If we take the first year in FaparLimitation.years as an integer then that
        #   provides an offset from zero for the indexing of the year sequence, and we
        #   can subtract that from the observation years to get the index of each
        #   observation.
        year_index = datetimes_year.astype("int") - fapar_limitation.years[0].astype(
            "int"
        )

        # Calculate daily mu value as m * daily molar assimilation:
        mu = fapar_limitation.lai_to_gpp_ratio_m[year_index] * daily_gpp

        # Calculate the Lambert W0 value
        daily_LAI = mu + (1 / self.phenology_const.k) * lambertw(
            -self.phenology_const.k * mu * np.exp(-self.phenology_const.k * mu), k=0
        )

        # Check that all imaginary parts are zero or np.nan
        if not np.all(np.logical_or(np.imag(daily_LAI) == 0, np.isnan(daily_LAI))):
            raise ValueError("Imaginary parts of Lambert W calculation are not zero")

        # Clip the real parts at zero
        daily_LAI = np.clip(np.real(daily_LAI), a_min=0, a_max=None)

        # Find the daily minimum of the lambert term and annual maximum LAI as the
        # steady state LAI
        self.steady_state_LAI = np.minimum(
            daily_LAI, fapar_limitation.lai_max[year_index]
        )
        """The steady state leaf area index for each day."""
        # Calculate the lagged value
        self.realised_LAI = exponential_moving_average(
            self.steady_state_LAI, alpha=alpha
        )
        """The realised leaf area index for each day given the modelled lag in responses
        to changing assimilation."""
