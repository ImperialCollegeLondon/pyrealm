"""The ``splash`` submodule provides the main SplashModel class for calculating
predictions under the SPLASH model.
"""  # noqa: D205

import warnings

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.bounds import BoundsChecker
from pyrealm.core.calendar import Calendar
from pyrealm.core.pressure import calculate_patm
from pyrealm.core.time_series import broadcast_time
from pyrealm.core.utilities import check_input_shapes
from pyrealm.core.xarray import ArrayType, get_common_dims, xarray_inputs
from pyrealm.splash.evap import DailyEvaporativeFluxes
from pyrealm.splash.solar import DailySolarFluxes


class SplashModel:
    r"""Fitting the SPLASH model.

    The SplashModel class calculates the predictions of the SPLASH v1.0 model
    :cite:p:`davis:2017a`. The input variables of latitude, elevation, temperature,
    precipitation and incoming solar radiation of observations are initially used to
    calculate solar and evaporative fluxes, which are stored in the ``solar`` and
    ``evap`` attributes as instances of :class:`~pyrealm.splash.solar.DailySolarFluxes`
    and :class:`~pyrealm.splash.evap.DailyEvaporativeFluxes`.

    There are two options for providing radiation inputs:

    * Sunshine fraction:  this was used in the original SPLASH implementation
      :cite:p:`davis:2017a` and is provided as the radiation input variable in - for
      example - the CRU climate datasets.

    * Shortwave radiation: the SPLASH v2 model :cite:p:`sandoval:2024a` updated the
      calculation of solar fluxes to use data on shortwave downwelling radiation. This
      is provided as a radiation input in many other datasets, such as FluxNET sites and
      ERA5.

    The inputs to a SplashModel are expected to be arrays with time varying along the
    first dimension. Other dimensions represent observations at sites on a particular
    date.  The ``dates`` argument is expected to be a Calendar object with the same
    length as the first dimension.    If xarray inputs are used, ``dates`` should also
    be initialised using xarray inputs to ensure that the time dimension is set
    correctly.

    The main use of the SplashModel object is then to calculate the expected actual
    evapotranspiration (AET), soil moisture and runoff across the time series:

    * The :meth:`~pyrealm.splash.splash.SplashModel.estimate_initial_soil_moisture`
      method can be used to estimate the initial soil moisture from the first year of
      data.

    * The :meth:`~pyrealm.splash.splash.SplashModel.calculate_soil_moisture` then takes
      the initial estimate of soil moisture in observed sites and iterates over the
      days, applying the
      :meth:`~pyrealm.splash.splash.SplashModel.estimate_daily_water_balance` method to
      calculate the daily water balance across the whole time series.

    Args:
        latitude: The latitude of observations
        elevation: The elevation of observations (m), also used to calculate atmospheric
            pressure.
        temperature: Air temperature (°C)
        precipitation: Precipitation (mm/day)
        dates: The dates of the time series
        sunshine_fraction: The sunshine fraction (0-1, unitless)
        shortwave_radiation: Downwelling shortwave radiation (W m-2).
        soil_capacity: The maximum soil moisture capacity, defaulting to 150 (mm)
        core_const: A core constants instance.
        bounds_checker: A bounds checker instance.
    """

    def __init__(
        self,
        latitude: ArrayType[np.floating],
        elevation: ArrayType[np.floating],
        temperature: ArrayType[np.floating],
        precipitation: ArrayType[np.floating],
        dates: Calendar,
        sunshine_fraction: ArrayType[np.floating] | None = None,
        shortwave_radiation: ArrayType[np.floating] | None = None,
        soil_capacity: ArrayType[np.floating] = np.array([150.0]),
        core_const: CoreConst = CoreConst(),
        bounds_checker: BoundsChecker = BoundsChecker(),
    ):
        # Declare, type and docstring attributes
        self.elevation: NDArray[np.floating]
        """The elevation of sites."""
        self.latitude: NDArray[np.floating]
        """The latitude of sites."""
        self.temperature: NDArray[np.floating]
        """The air temperature in °C of daily observations."""
        self.precipitation: NDArray[np.floating]
        """The precipitation in mm of daily observations."""
        self.soil_capacity: NDArray[np.floating]
        """The maximum soil water capacity for sites (mm)."""
        self.dates: Calendar = dates
        """The dates of observations along the first array axis."""
        self.sunshine_fraction: NDArray[np.floating] | None = None
        """The sunshine fraction (0-1) of daily observations."""
        self.shortwave_radiation: NDArray[np.floating] | None = None
        """The downwelling shortwave radiation of daily observations (W m-2)."""

        # Handle sunshine fraction vs shortwave radiation
        if (sunshine_fraction is None) == (shortwave_radiation is None):
            raise ValueError("Provide one of sunshine_fraction or shortwave_radiation")

        # Get a single array object for validation
        if sunshine_fraction is not None:
            radiation_input = sunshine_fraction
        elif shortwave_radiation is not None:
            radiation_input = shortwave_radiation

        # Ensure first dimension is time if dates is also initialised with xarray
        self.dims = get_common_dims(
            elevation,
            latitude,
            temperature,
            precipitation,
            radiation_input,
            init_dims=dates.dims,
        )
        # Convert array inputs to numpy
        (elevation, latitude, temperature, precipitation, radiation_input) = (
            xarray_inputs(
                elevation,
                latitude,
                temperature,
                precipitation,
                radiation_input,
                dims=self.dims,
            )
        )

        # Check input sizes are congurent
        self.shape: tuple = check_input_shapes(
            elevation, latitude, temperature, precipitation, radiation_input
        )
        """The array shape of the input variables"""

        if self.shape[0] == 1:
            self.shape = (len(dates), *self.shape[1:])
        elif self.shape[0] != len(dates):
            raise ValueError(
                "The first dimension of inputs must either match the number of dates or"
                " have a length of one."
            )

        # Broadcast all the inputs over time to simplify the daily indexing if any
        # inputs are constant over time
        elevation = broadcast_time(elevation, self.shape)
        latitude = broadcast_time(latitude, self.shape)
        temperature = broadcast_time(temperature, self.shape)
        precipitation = broadcast_time(precipitation, self.shape)
        radiation_input = broadcast_time(radiation_input, self.shape)

        self.elevation = bounds_checker.check("elevation", elevation)
        self.latitude = bounds_checker.check("latitude", latitude)
        self.temperature = bounds_checker.check("tc", temperature)
        self.precipitation = bounds_checker.check("precipitation", precipitation)
        self.kWm = bounds_checker.check("soil_capacity", soil_capacity)

        # Assign radiation variable back to the appropriate attribute
        if sunshine_fraction is not None:
            self.sunshine_fraction = bounds_checker.check(
                "sunshine_fraction", radiation_input
            )
        else:
            self.shortwave_radiation = bounds_checker.check(
                "shortwave_radiation", radiation_input
            )

        # TODO - potentially allow _actual_ climatic pressure data as an input
        self.patm: NDArray[np.floating] = calculate_patm(
            elevation=elevation, core_const=core_const
        )
        """The atmospheric pressure at sites, derived from elevation"""

        # Calculate the daily solar fluxes - these are invariant across the simulation
        self.solar: DailySolarFluxes = DailySolarFluxes(
            latitude=self.latitude,
            elevation=self.elevation,
            dates=dates,
            sunshine_fraction=self.sunshine_fraction,
            shortwave_radiation=self.shortwave_radiation,
            temperature=self.temperature,
            core_const=core_const,
        )
        """Estimated solar fluxes for observations"""

        # Initialise the evaporative flux class
        self.evap: DailyEvaporativeFluxes = DailyEvaporativeFluxes(
            solar=self.solar,
            patm=self.patm,
            temperature=self.temperature,
            core_const=core_const,
        )
        """Estimated evaporative fluxes for observations"""

    def estimate_initial_soil_moisture(
        self,
        initial_soil_moisture: ArrayType[np.floating] | None = None,
        max_iter: int = 10,
        max_diff: float = 1.0,
        return_convergence: bool = False,
        verbose: bool = False,
    ) -> NDArray[np.floating]:
        """Estimate initial soil moisture.

        This method uses the first year of data provided to a SplashModel instance to
        estimate initial values for the soil moisture data. The process assumes that the
        soil moisture cycle is stationary over the first year of data and iteratively
        updates an initial guess at start of year soil moisture until those values are
        within a given tolerance of the end of year values. The method cannot be run
        when less than one year of data is provided to the model.

        The user can provide an array of initial values across sites, defaulting to an
        initial guess of zero soil moisture in all sites . The user can also control the
        maximum number of update iterations and the accepted tolerance for convergence.
        The method will normally fail if the estimates do not converge, but the
        ``return_convergence`` option can be used to return the estimated soil moisture
        at each iteration regardless of the success of convergence.

        Args:
            initial_soil_moisture: An optional estimate of initial soil moisture.
            max_iter: The maximum number of iterations used to achieve convergence.
            max_diff: The maximum acceptable difference between year start and year end
                soil moisture,
            return_convergence: Optionally return an array of soil moistures at the end
                of each iteration even when convergence fails.
            verbose: Optionally turn on detailed logging of the iteration process.

        Returns:
            An array of the estimated starting soil moisture. If ``return_convergence``
            is set to True, the returned array will have an additional dimension for
            each iteration of the equilibration loop.

        Raises:
            ValueError: The input data are of the wrong shape, contain invalid values or
                do not include at least a full year of data.
            RuntimeError: The estimation fails to converge within the set number of
                iterations.
        """

        # Initialise loop termination
        equilibrated = False
        n_iter = 0
        wn_ret = []

        if initial_soil_moisture is not None:
            wn_init = xarray_inputs(initial_soil_moisture, dims=self.dims[1:])
            # Check the shape is the same as the shape of a slice along axis 0
            if wn_init.shape != self.shape[1:]:
                raise ValueError("Incorrect shape in wn_init")
            if np.any((wn_init < 0) | (wn_init > self.kWm)):
                raise ValueError(
                    "Soil moisture must be between 0 and the soil capacity."
                )
            wn_start = wn_init
        else:
            wn_start = np.zeros(self.shape[1:])

        # Find a date one year into the future from the first calendar date.
        date_start = pd.Timestamp(self.dates[0].date)
        date_end = date_start + pd.DateOffset(years=1)
        num_days = (date_end - date_start).days

        if len(self.dates) < num_days:
            raise ValueError("Cannot equilibrate - less than one year of data")

        # Run the equilibration loop
        while (not equilibrated) and (n_iter < max_iter):
            # Track the iterations
            n_iter += 1

            # Loop over the calendar object, updating the soil_moisture array
            wn_day = wn_start
            for day_idx in range(num_days):
                # Calculate aet, soil moisture and runoff:
                _, wn_day, _ = self.estimate_daily_water_balance(
                    previous_soil_moisture=wn_day, day_index=day_idx
                )

            # Calculate the difference between the start of year soil moisture and the
            # final day of the year and then update the start point to the end of year.
            diff_sm = np.abs(wn_start - wn_day)
            cur_diff = np.nanmax(diff_sm)
            wn_start = wn_day

            # Optionally store the soil moisture at the end of each loop
            if return_convergence:
                wn_ret.append(wn_start)

            # Report if verbose
            if verbose:
                print(f"Iteration: {n_iter}; maximum difference: {cur_diff}")

            if cur_diff <= max_diff:
                equilibrated = True

        # Check for convergence failure before returning the final values.
        if not equilibrated:
            msg = (
                f"Initial soil moisture did not converge within {n_iter} iterations:"
                f"maximum absolute difference = {cur_diff}"
            )
            if return_convergence:  # always returns without raising an error
                warnings.warn(msg)
            else:
                raise RuntimeError(msg)

        if return_convergence:  # returns the values of wn at each iteration
            return np.array(wn_ret)  # (n_iter, *wn_start.shape)
        else:
            return wn_start

    def estimate_daily_water_balance(
        self,
        previous_soil_moisture: ArrayType[np.floating],
        day_index: int | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        r"""Estimate the daily water balance.

        This function estimates the daily water balance within observations. The
        function first calculates the expected actual evapotranspiration (mm d-1,
        :math:`\textrm{AET}_{[t]}`), given the soil moisture from the preceding day
        (mm, :math:`W_{n[t-1]}`). Those are then used, along with the precipitation (mm
        d-1, :math:`P_{[t]}`) and condensation (mm d-1, :math:`C_{[t]}`) for the current
        day, to calculate the current soil moisture (mm, :math:`W_{n[t]}`) as:

        .. math::

            W_{n[t]} = W_{n[t-1]} + P_{[t]} + C_{[t]} - \textrm{AET}_{[t]}.

        When the resulting soil moisture exceeds the maximum capacity of the soil
        (:attr:{soil_capacity}, :math:`W_m`), the excess is allocated to run off,
        leaving the soil saturated. Note that the soil moisture is not altered by
        subsurface flow: there is not vertical or horizontal transfer of water from the
        soil, only losses through evapotranspiration. Negative soil moisture values are
        replaced by zero.

        By default, ``previous_soil_moisture`` is expected to provide estimates for all
        observations across all days in the model, but ``day_index`` can be set to
        provide an estimate for only one particular day, for use in iterating over time
        series.

        Args:
            previous_soil_moisture: Soil moisture estimates for the preceding day (mm)
            day_index: Optionally, the index of the date for which to calculate water
                balance.

        Returns:
            A tuple of numpy arrays containing estimated  AET, daily soil moisture and
            runoff.
        """

        # Check day_idx inputs to map either the single time index given in day_idx or
        # the whole dataset.
        if day_index is None:
            splash_dims = self.dims
            splash_shape = self.shape
            didx: int | slice = slice(self.shape[0])
        else:
            splash_dims = self.dims[1:]
            splash_shape = self.shape[1:]
            didx = day_index

        previous_wn = xarray_inputs(previous_soil_moisture, dims=splash_dims)
        try:
            check_input_shapes(previous_wn, shape=splash_shape)
        except ValueError:
            msg = (
                "The shape of previous_wn does not match the existing SPLASH model data"
            )
            raise ValueError(msg)

        # Calculate the expected aet_d given the previous wn
        if np.any((previous_wn < 0) | (previous_wn > self.kWm)):
            raise ValueError("Soil moisture must be between 0 and kWm")

        aet = self.evap.estimate_aet(soil_moisture=previous_wn, day_index=day_index)

        # Calculate current soil moisture, mm
        current_wn = (
            previous_wn + self.precipitation[didx] + self.evap.condensation[didx] - aet
        )

        # Partition current_wn into soil moisture and runoff (ro), mm
        # - allocate excess sm to runoff and clip out negative sm
        ro = np.clip(current_wn, self.kWm, np.inf) - self.kWm
        wn = np.clip(current_wn, 0, self.kWm)

        # Return values, ignoring the type clash that estimate_aet _can_ return
        # additional arrays. aet here is explicitly a single array not a tuple.
        return aet, wn, ro  # type: ignore

    def calculate_soil_moisture(
        self,
        initial_soil_moisture: ArrayType[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Calculate the soil moisture, AET and runoff from a SplashModel.

        This function takes an initial array of soil moisture values for the first
        observations in a SplashModel time series and then iteratively applies the daily
        water balance calculations along the time axis using the
        :meth:`~pyrealm.splash.splash.SplashModel.estimate_daily_water_balance` method.
        This produces the expected actual evapotranspiration (AET), soil moisture,
        runoff and for all sites across the time series.

        Args:
            initial_soil_moisture: Estimated initial soil moisture for observations.

        Returns:
            A tuple of numpy arrays containing predicted AET, soil moisture and runoff.
        """

        wn_init = xarray_inputs(initial_soil_moisture, dims=self.dims[1:])
        try:
            check_input_shapes(wn_init, shape=self.shape[1:])
        except ValueError:
            raise ValueError(
                "The shape of initial_soil_moisture does not match the "
                "existing SPLASH model data"
            )

        # Create storage for outputs
        aet_out = np.full(self.shape, np.nan)
        wn_out = np.full(self.shape, np.nan)
        ro_out = np.full(self.shape, np.nan)

        if np.any((wn_init < 0) | (wn_init > self.kWm)):
            raise ValueError("Soil moisture must be between 0 and kWm")

        curr_wn = wn_init

        for day_idx in range(self.shape[0]):
            # Calculate the balance for this date, updating the input for
            # the following day
            aet, curr_wn, ro = self.estimate_daily_water_balance(
                previous_soil_moisture=curr_wn, day_index=day_idx
            )

            # Convert the outputs to scalars if there is only a time axis
            if len(self.shape) == 1:
                aet = aet.squeeze()
                curr_wn = curr_wn.squeeze()
                ro = ro.squeeze()

            # Store the outputs to return
            aet_out[day_idx] = aet
            wn_out[day_idx] = curr_wn
            ro_out[day_idx] = ro

        return aet_out, wn_out, ro_out
