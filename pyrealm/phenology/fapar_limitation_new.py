"""Class to compute the fAPAR_max and annual peak Leaf Area Index (LAI).

The :class:`FaparLimitationNew` class and the :meth:`FaparLimitationNew.from_pmodel` are
designed to work with inputs that can have multiple dimensions. The first axis is
_always_ assumed to represent a time series of annual observations. If the inputs are
one dimensional, then this is a time series for a single site; if they are three
dimensional then these are observations for a grid of sites. Usually all array inputs
will have the same shape but note the following instances where you might need to take
care with array broadcasting.

* Growing season length might well be constant across sites. If so - for example
    with 3D data - the input would need shape `(N, 1, 1)` to broadcast N years of
    data over the array of sites.
* The climatological aridity index is very likely to be constant through time.
    If so, the array should have a singleton first dimension to broadcast an
    observation per site across observations. For example, with `(10, 3, 3)` data
    (10 years for a 3x3 grid of sites), the aridity index could be provided as
    `(1,3,3)` to broadcast the aridity index across each year. It could also use a
    single scalar value to use the same aridity index for all sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants.phenology_const import PhenologyConstNew
from pyrealm.core.experimental import warn_experimental
from pyrealm.core.time_series import AnnualValueCalculator
from pyrealm.core.utilities import (
    check_input_shapes,
    summarize_attrs,
)
from pyrealm.pmodel.pmodel import PModel, PModelABC, SubdailyPModel

FAPAR_LIMITATION_METHOD_CLASS_REGISTRY: dict[str, type[FaparLimitationMethodABC]] = {}
"""A registry for optimal chi calculation classes.

Different implementations of the calculation of optimal chi must all be subclasses of
:class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC` abstract base class.
This dictionary is used as a registry for defined subclasses and a method name
is used to retrieve a particular implementation from this registry. For example:

.. code:: python

    prentice14_opt_chi = OPTIMAL_CHI_CLASS_REGISTRY['prentice14']
"""


class FaparLimitationMethodABC(ABC):
    r"""Abstract base class for methods computing maximum annual fAPAR and LAI."""

    __experimental__ = True

    method: str
    """A short method name used to identify the class in
    :data:`FAPAR_LIMITATION_METHOD_CLASS_REGISTRY`.
    """

    attrs: tuple[tuple[str, str], ...]
    """A tuple of attributes to be reported in the FaparLimitation.summarize() output
    for the method."""

    requires: tuple[str, ...]
    """A tuple of any additional variables required to create a FaparLimitation
    instance using the method."""

    def __init__(
        self, fapar_limitation: FaparLimitationNew, phenology_const: PhenologyConstNew
    ):
        """Initialise the method instance.

        Subclass should provide a method specific instance that calls
        `super().__init__(...)`, carries out method specific validation and then runs
        ``self.set_z_and_f0``.
        """
        self.fapar_limitation = fapar_limitation
        self.phenology_const = phenology_const

        self.f0: NDArray[np.floating]
        self.z: NDArray[np.floating]

        # Check for required variables
        self._check_required_variables()

    def _check_required_variables(self) -> None:
        """Check required variables.

        Checks that any required variables for the method have been passed to the
        FaparLimitation constructor.
        """
        for var in self.requires:
            if not hasattr(self.fapar_limitation, var):
                raise ValueError(
                    f"Values for {var} are required to use the {self.method} method "
                    "with FaparLimitation."
                )

    @abstractmethod
    def set_z_and_f0(self) -> None:
        """Sets the f0 and z values for the method."""
        pass

    @abstractmethod
    def calculate_maximum_fapar(
        self, energy_limited_fapar: NDArray, water_limited_fapar: NDArray
    ) -> NDArray[np.floating]:
        """Calculate the maximum fAPAR.

        Provides the method specific calculation of maximum fAPAR from the energy and
        water limited maximum values and should return the calculated maximum fAPAR.

        Args:
            energy_limited_fapar: The maximum fAPAR given energy limitation.
            water_limited_fapar: The maximum fAPAR given water limitation.
        """
        pass

    @classmethod
    def __init_subclass__(cls, method: str) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        FAPAR_LIMITATION_METHOD_CLASS_REGISTRY[cls.method] = cls


class FaparLimitationMethodCai(FaparLimitationMethodABC, method="cai"):
    r"""Compute maximum annual fAPAR and LAI following :cite:t:`cai:2025a`.

    The method of :cite:t:`cai:2025a` uses a single global value of :math:`z` but models
    :math:`f_0` as a function of site-specific aridity, expressed as the climatological
    ratio PET/P, (see :meth:`set_z_and_f0`). The annual maximum fAPAR is then the simple
    minimum of the site values for water and energy limited fAPAR.
    """

    __experimental__ = True

    attrs = (
        ("lai_max", "-"),
        ("fapar_max", "-"),
    )

    requires = ("aridity_index",)

    def __init__(
        self, fapar_limitation: FaparLimitationNew, phenology_const: PhenologyConstNew
    ):
        """Initialise a FaparLimitationMethod instance using the Cai approach."""

        # Run the superclass init method.
        super().__init__(
            fapar_limitation=fapar_limitation, phenology_const=phenology_const
        )

        # This is only set as a side effect of the calculate_maximum_fapar method being
        # called, which is a little bit hacky, but at the moment just preserving the
        # attribute _somewhere_. Will see how this class evolves.
        self.energy_limited: NDArray[np.bool_]
        """Boolean array showing if annual :math:`fAPAR_{max}` is water or energy
        limited."""

        # Validate the aridity index
        self.aridity_index = getattr(self.fapar_limitation, "aridity_index")

        # Check aridity index is a site specific array (or constant)
        try:
            np.broadcast_shapes(
                self.aridity_index[None, :].shape, self.fapar_limitation.shape
            )
        except ValueError:
            raise ValueError(
                f"The aridity index argument must contain site specific values and "
                f"must be a constant of have shape {self.fapar_limitation.shape[1:]}"
            )

        # Make sure the aridity index is not zero
        if np.any(self.aridity_index <= 0):
            raise ValueError("The aridity index has to be positive.")

        # Set z and f0
        self.set_z_and_f0()

    def set_z_and_f0(self) -> None:
        r"""Set the :math:`z` and :math:`f_0` parameters.

        The value :math:`f_0` is the ratio of annual total transpiration of annual total
        precipitation. It is calculated from site specific estimates of the
        climatological aridity index, calculated as the long term (typically 20 years)
        total PET over total precipitation (:math:`AI`, unitless) as:

        .. math::

                f_0 = a \exp{\left(-b \left(\frac{AI}{c}\right)^2\right)}

        where :math:`a,b,c` are defined in the
        :attr:`~pyrealm.constants.phenology_const.PhenologyConstNew.cai_f0_coefficients`
        attribute.
        """

        a, b, c = self.phenology_const.cai_f0_coefficients

        self.f0 = a * np.exp(-b * np.log(self.aridity_index / c) ** 2)
        self.z = np.array([self.phenology_const.cai_z])

    def calculate_maximum_fapar(
        self, energy_limited_fapar: NDArray, water_limited_fapar: NDArray
    ) -> NDArray:
        """Calculate the maximum fAPAR.

        The Cai method uses the simple minimum of the water and energy limited fAPAR
        values as the maximum possible fAPAR.

        Args:
            energy_limited_fapar: The maximum fAPAR given energy limitation.
            water_limited_fapar: The maximum fAPAR given water limitation.
        """

        # Calculate fAPAR max and record whether the location is energy or water limited
        fapar_max = np.minimum(water_limited_fapar, energy_limited_fapar)
        self.energy_limited = energy_limited_fapar < water_limited_fapar

        return fapar_max


class FaparLimitationMethodZhu(FaparLimitationMethodABC, method="zhu"):
    r"""Compute maximum annual fAPAR and LAI following :cite:t:`zhu:2026a`.

    The method of :cite:t:`zhu:2026a` uses single global values of both :math:`z` and
    :math:`f_0`. However, the annual maximum fAPAR is then a continuous function of
    water and energy limited fAPAR following a Budyko curve (see
    :meth:`calculate_maximum_fapar` for details).
    """

    __experimental__ = True

    attrs = (
        ("lai_max", "-"),
        ("fapar_max", "-"),
    )

    requires = tuple()

    def __init__(
        self, fapar_limitation: FaparLimitationNew, phenology_const: PhenologyConstNew
    ):
        """Initialise a FaparLimitationMethod instance using the Zhu approach."""

        # Run the superclass init method.
        super().__init__(
            fapar_limitation=fapar_limitation, phenology_const=phenology_const
        )

        # Set z and f0
        self.set_z_and_f0()

    def set_z_and_f0(self) -> None:
        r"""Set the :math:`z` and :math:`f_0` parameters.

        This method has fixed values for :math:`z` (see
        :attr:`~pyrealm.constants.phenology_const.PhenologyConstNew.zhu_f0`) and
        :math:`f_0` (see
        :attr:`~pyrealm.constants.phenology_const.PhenologyConstNew.zhu_z`)
        """

        self.f0 = np.array([self.phenology_const.zhu_f0])
        self.z = np.array([self.phenology_const.zhu_z])

    def calculate_maximum_fapar(
        self, energy_limited_fapar: NDArray, water_limited_fapar: NDArray
    ) -> NDArray:
        r"""Calculate the maximum fAPAR.

        The Zhu method calculates the maximum fAPAR as a function of the energy and
        water limited fAPAR, following a function like the Budyko curve
        :cite:p:`roderick:2011a`:
    
        .. math::
            :nowrap:

            \[    
                \begin{align*}

                    c_w &= \frac{f_{APAR_{c}}}{f_{APAR_{w}}} \\
                    f_{APAR_{max}} = f_{APAR_{w}} \left(1 + c_w - 
                    \left( 1 + c_w ^{\bar\omega} \right)^{1/{\bar\omega}}

                \end{align*}
            \],

        where :math:`\bar\omega` is defined in the
        :attr:`~pyrealm.constants.phenology_const.PhenologyConstNew.zhu_budyko`
        constants parameter.

        Args:
            energy_limited_fapar: The maximum fAPAR given energy limitation.
            water_limited_fapar: The maximum fAPAR given water limitation.
        """
        # Calculate the ratio of energy limited to water limited fAPAR, safeguarding
        # against divide by zero
        cw_ratio = energy_limited_fapar / (
            np.clip(water_limited_fapar, a_min=np.finfo(float).eps, a_max=None)
        )

        # Calculate the maximum fapar.
        fapar_max = (
            (1 + cw_ratio)
            - (1 + cw_ratio**self.phenology_const.zhu_budyko)
            ** (1 / self.phenology_const.zhu_budyko)
        ) * water_limited_fapar

        return fapar_max


class FaparLimitationNew:
    r"""Compute maximum annual fAPAR and LAI.

    This class calculates maximum annual fAPAR, which can be limited either by the
    availability of light energy ($f_{APAR_{c}}$) or by the availability of water
    ($f_{APAR_{w}}$). The equations for these two variables, following :cite:`cai:2025a`
    are:

    .. math::
        :nowrap:

        \[    
          \begin{align*}
            f_{APAR_{c}} &= 1 - \frac{z}{k A_0}\\
            f_{APAR_{w}} &= \left(\frac{ c_a \left( 1 - \chi \right)}{ 1.6 D }\right)
                            \left(\frac{ f_0 P }{ A_0 }\right) \\
          \end{align*}
        \]

    The maximum fAPAR is then calculated as a function of :math:`f_{APAR_{c}}` and
    :math:`f_{APAR_{w}}`. In these equations:

    * :math:`z` accounts for the growth and maintenance costs of leaves.
    * :math:`f_0` accounts for water limitation on annual assimulation and is is the
      ratio of annual total transpiration to annual total precipitation.

    The other variables are the required arguments to the class defined below. 
    
    There are different approaches to estimating :math:`z` and :math:`f_0` and to
    calculating the maximum fAPAR from the two inputs. For details see:

    * ``method=cai``; :class:`FaparLimitationMethodCai`
    * ``method=zhu``; :class:`FaparLimitationMethodZhu`

    The maximum annual LAI can then be calculated using Beer's law as:

    .. math::

        L_{max} = - ( 1 / k ) \ln {1 -f_{APAR_{max}}}

    The most common source of most of the variables needed to calculate maximum fAPAR is
    a P Model, and the
    :meth:`~pyrealm.phenology.fapar_limitation.FaparLimitation.from_pmodel` method can
    be used to estimate maximum fAPAR directly from a fitted P Model.

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
        years: An array of year datetimes for the observations.
        method: The method to be applied when calculating maximum fAPAR, defaulting to
        ``cai``. 
        phenology_const: An instance of
            :class:`~pyrealm.constants.phenology_const.PhenologyConstNew`
        **kwargs: Any additional variables required by specific method choices.
    """

    __experimental__ = True

    def __init__(
        self,
        annual_total_potential_gpp: NDArray[np.floating],
        annual_mean_ca: NDArray[np.floating],
        annual_mean_chi: NDArray[np.floating],
        annual_mean_vpd: NDArray[np.floating],
        annual_total_precip: NDArray[np.floating],
        annual_growing_season_length: NDArray[np.floating],
        years: NDArray[np.datetime64],
        method: str = "cai",
        phenology_const: PhenologyConstNew = PhenologyConstNew(),
        **kwargs: NDArray[np.floating],
    ) -> None:
        # Experimental class
        warn_experimental("FaparLimitation")

        # Validate the input shapes.
        # NOTE: Input shape checking only handles the main arguments and not the
        #       contents of kwargs arrays that are passed down as required extra
        #       arguments to FaparLimitationMethodABC subclasses. This is because those
        #       required variables can have different expectations of the inputs shapes
        #       (e.g. FaparLimitationMethodCai requires aridity_index for sites not
        #       across time), and so shape of those inputs should be validated within
        #       subclasses, not here.
        self.shape: tuple[int, ...] = check_input_shapes(
            annual_total_potential_gpp,
            annual_mean_ca,
            annual_mean_chi,
            annual_mean_vpd,
            annual_total_precip,
            annual_growing_season_length,
        )

        # Check the years values - must be datetime64[Y] and be one dimensional,
        # matching the first dimension of the other inputs
        # TODO - this is a bit stringent, but is more robust
        if not years.dtype == "<M8[Y]":
            raise ValueError("The years argument must provide np.datetime64[Y] values")

        if years.shape != (self.shape[0],):
            raise ValueError(
                "The years argument must be one dimensional and match the length "
                "of the first axis of the other arguments"
            )

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

        # Additional variables -  add them to the instance
        for var_name, var_values in kwargs.items():
            setattr(self, var_name, var_values)

        self._additional_vars: tuple[str, ...] = tuple(kwargs.keys())
        """A tuple containing the attribute names of additional variables passed to the
        FaparLimitation instance."""

        # Constants used for phenology computations
        self.phenology_const: PhenologyConstNew = phenology_const

        # Set up the fAPAR limitation method to be used.
        self.method: str = method

        if method not in FAPAR_LIMITATION_METHOD_CLASS_REGISTRY:
            raise ValueError(f"Unknown FaparLimitation method: {method}")

        # Get an instance of the method class from the registry
        self.limitation_method: FaparLimitationMethodABC = (
            FAPAR_LIMITATION_METHOD_CLASS_REGISTRY[method](
                fapar_limitation=self, phenology_const=self.phenology_const
            )
        )

        # Calculate the energy and water limited terms.
        energy_limited_fapar = 1.0 - self.limitation_method.z / (
            self.phenology_const.k * annual_total_potential_gpp
        )
        water_limited_fapar = (
            self.limitation_method.f0
            * annual_total_precip
            * annual_mean_ca
            * (1 - annual_mean_chi)
            / (1.6 * annual_mean_vpd * annual_total_potential_gpp)
        )

        # Calculate the maximum fapar using the limitation method
        self.fapar_max: NDArray[np.floating] = (
            self.limitation_method.calculate_maximum_fapar(
                energy_limited_fapar=energy_limited_fapar,
                water_limited_fapar=water_limited_fapar,
            )
        )

        # """Estimated annual maximum fAPAR (unitless)."""
        # self.energy_limited: NDArray[np.bool_] = fapar_energylim < fapar_waterlim
        # """Boolean array showing if annual :math:`fAPAR_{max}` is water or energy
        # limited."""

        self.lai_max: NDArray[np.floating] = -(1 / self.phenology_const.k) * np.log(
            1.0 - self.fapar_max
        )
        """Estimated annual maximum LAI (unitless)"""

        # self.lai_to_gpp_ratio_m = (
        #     self.phenology_const.sigma
        #     * self.annual_growing_season_length
        #     * self.lai_max
        # ) / (self.annual_total_potential_gpp * self.fapar_max)
        # """The steady state ratio of leaf area index to potential GPP (:math:`m`)"""

    def __repr__(self) -> str:
        """Simple representation of class instance."""
        return f"FaparLimitationNew(shape={self.shape})"

    def summarize(self, dp: int = 2) -> None:
        """Print summary of estimates of fAPAR limitation.

        Prints a summary of the calculated values in a FaparLimitation instance
        including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        summarize_attrs(self, self.limitation_method.attrs, dp=dp)

    @classmethod
    def from_pmodel(
        cls,
        pmodel: PModelABC,
        growing_season: NDArray[np.bool],
        precip: NDArray[np.floating],
        datetimes: NDArray[np.datetime64] | None = None,
        method: str = "cai",
        phenology_const: PhenologyConstNew = PhenologyConstNew(),
        **kwargs: NDArray[np.floating],
    ) -> FaparLimitationNew:
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
          provide an array of boolean values indicating which observations should be
          treated as in the growing season.

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

        .. NOTE:

            Potential GPP is taken directly from the P Model instance. If you want to
            apply a post-hoc penalty factor to GPP (e.g. a water limitation factor),
            then use the `apply_gpp_penalty_factor` method to your fitted P Model before
            using it to calculate fAPAR limitation.

        Args:
            pmodel: A :class:`pyrealm.pmodel.pmodel.PModel` or
                :class:`pyrealm.pmodel.pmodel.SubdailyPModel` instance, fitted with
                ``fapar`` fixed at one.
            datetimes: An array giving the datetimes of observations.
            growing_season: A boolean array indicating which observations are to be
                considered as part of the growing season.
            precip: An array of precipitation for each observation.
            method: The method to be used in calculating maximum fAPAR, defaulting to
                `cai`.
            phenology_const: An instance of
                :class:`~pyrealm.constants.phenology_const.PhenologyConstNew`
            **kwargs: Any additional variables required by specific method choices.
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

        else:
            raise TypeError("Invalid PModel class")

        # Ensure data_shape includes the full time dimension
        n_time = len(datetimes) if pmodel.shape[0] == 1 else pmodel.shape[0]
        data_shape = (n_time, *pmodel.shape[1:])

        # Create the annual value calculator
        # - the code above guards against datetimes being None
        avc = AnnualValueCalculator(
            data_shape=data_shape,
            timing=datetimes,  # type: ignore [arg-type]
            subset_mask=growing_season,
        )

        # Inputs to avc.get_annual_means/totals need the full shapes, so first broadcast
        gpp = np.broadcast_to(pmodel.gpp, data_shape)
        ca = np.broadcast_to(pmodel.env.ca, data_shape)
        chi = np.broadcast_to(pmodel.optchi.chi, data_shape)
        vpd = np.broadcast_to(pmodel.env.vpd, data_shape)
        precip = np.broadcast_to(precip, data_shape)

        # Get the total GPP for each observation
        # TODO: handle incompleteness - when do we stop estimating annual values from
        #         partial years (or at least warn about it)

        # Calculate annual mean potential GPP and scale up to the year - note that this
        # includes any post-hoc GPP penalty applied to the model
        annual_mean_potential_gpp = avc.get_annual_means(gpp)
        annual_total_potential_gpp = (
            annual_mean_potential_gpp * (avc.year_n_days) * 86400 * 1e-6
        ) / pmodel.core_const.k_c_molmass

        # Calculate annual mean ca, chi and VPD within growing season
        annual_mean_ca = avc.get_annual_means(ca, within_subset=True)
        annual_mean_chi = avc.get_annual_means(chi, within_subset=True)
        annual_mean_vpd = avc.get_annual_means(vpd, within_subset=True)

        # Calculate total annual precipitation
        annual_total_precip = avc.get_annual_totals(precip)

        return cls(
            annual_total_potential_gpp=annual_total_potential_gpp,
            annual_mean_ca=annual_mean_ca,
            annual_mean_chi=annual_mean_chi,
            annual_mean_vpd=annual_mean_vpd,
            annual_total_precip=annual_total_precip,
            annual_growing_season_length=avc.year_n_days_subset,
            years=avc.years,
            method=method,
            phenology_const=phenology_const,
            **kwargs,
        )
