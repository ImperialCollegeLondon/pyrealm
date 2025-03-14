"""The :mod:`~pyrealm.pmodel.functions` submodule contains the main standalone functions
used for calculating the photosynthetic behaviour of plants. The documentation describes
the key equations used in each function.
"""  # noqa D210, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.core.utilities import check_input_shapes
from pyrealm.core.water import calc_viscosity_h2o


def calculate_simple_arrhenius_factor(
    tk: NDArray[np.float64],
    tk_ref: float,
    ha: float,
    k_R: float = CoreConst().k_R,
) -> NDArray[np.float64]:
    r"""Calculate an Arrhenius scaling factor using activation energy.

    Calculates the temperature-scaling factor :math:`f` for enzyme kinetics following
    a simple Arrhenius response governed solely by the activation energy for an enzyme
    (``ha``, :math:`H_a`). The rate is given for a temperature :math:`T` relative to a
    reference temperature :math:T_0`, both given in Kelvin.

    Arrhenius kinetics are described as:

    .. math::

        x(T) = \exp(c - H_a / (T R))

    The temperature-correction function :math:`f(T, H_a)` is:

      .. math::
        :nowrap:

        \[
            \begin{align*}
                f &= \frac{x(T)}{x(T_0)} \\
                  &= \exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)
                        \text{, or equivalently}\\
                  &= \exp \left( \frac{ H_a}{R} \cdot
                        \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
            \end{align*}
        \]

    Args:
        tk: Temperature (K)
        tk_ref: The reference temperature for the reaction (K).
        ha: Activation energy (in :math:`J \text{mol}^{-1}`)
        k_R: The universal gas constant, defaulting to the value from
            attr:`~pyrealm.constants.core_const.CoreConst.k_R`. 
    
    Examples:
        >>> # Percentage rate change from 25 to 10 degrees Celsius
        >>> at_10C = calculate_simple_arrhenius_factor(
        ...     np.array([283.15]) , 298.15, 100000
        ... )
        >>> np.round((1.0 - at_10C) * 100, 4)
        array([88.1991])
    """

    return np.exp(ha * (tk - tk_ref) / (tk_ref * k_R * tk))


def calculate_kattge_knorr_arrhenius_factor(
    tk_leaf: NDArray[np.float64],
    tk_ref: float,
    tc_growth: NDArray[np.float64],
    coef: dict[str, float],
    k_R: float = CoreConst().k_R,
) -> NDArray[np.float64]:
    r"""Calculate an Arrhenius factor following :cite:t:`Kattge:2007db`.

    This implements a "peaked" version of the Arrhenius relationship, describing a
    decline in reaction rates at higher temperatures. In addition to the activation
    energy (see :meth:`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`),
    this implementation adds an entropy term and the deactivation energy of the enzyme
    system. The rate is given for a given instantaneous temperature :math:`T` relative
    to a reference temperature :math:T_0`, both given in Kelvin, but the entropy is
    calculated using a separate estimate of the growth temperature for a plant,
    expressed in °C.


    .. math::
        :nowrap:

        \[
            \begin{align*}

                f  &= \exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)
                      \left(
                        \frac{1 + \exp \left( \frac{T_0 \Delta S - H_d }{ R T_0}\right)}
                             {1 + \exp \left( \frac{T \Delta S - H_d}{R T} \right)}
                      \right)
                      \left(\frac{T}{T_0}\right)
            \end{align*}

            \text{where,}

            \Delta S = a + b * t_g

        \]

    The coefficients dictionary must provide entries for:

    * ha: The activation energy of the enzyme (:math:`H_a`)
    * hd: The deactivation energy of the enzyme (:math:`H_d`)
    * entropy_intercept: The intercept of the entropy relationship (:math:`a`)
    * entropy_slope: The slope of the entropy relationship (:math:`b`)

    Args:
        tk_leaf: The instantaneous temperature in Kelvin (K) at which to calculate the
            factor (:math:`T`)
        tk_ref: The reference temperature in Kelvin for the process (:math:`T_0`)
        tc_growth: The growth temperature of the plants in °C (:math:`t_g`)
        coef: A dictionary providing values of the coefficients ``ha``,
            ``hd``, ``entropy_intercept`` and ``entropy_slope``.
        k_R: The universal gas constant, defaulting to the value from
            attr:`~pyrealm.constants.core_const.CoreConst.k_R`.

    Returns:
        Values for :math:`f`

    Examples:
        >>> # Calculate the factor for the relative rate of Vcmax at 10 °C (283.15K)
        >>> # compared to the rate at the reference temperature of 25°C (298.15K).
        >>> from pyrealm.constants import PModelConst
        >>> pmodel_const = PModelConst()
        >>> # Get enzyme kinetics parameters
        >>> coef = pmodel_const.arrhenius_vcmax['kattge_knorr']
        >>> # Calculate the arrhenius factor
        >>> val = calculate_kattge_knorr_arrhenius_factor(
        ...     tk_leaf= np.array([283.15]),
        ...     tc_growth = 10,
        ...     tk_ref=298.15,
        ...     coef=coef,
        ... )
        >>> np.round(val, 4)
        array([0.261])
    """

    # Calculate entropy as a function of temperature _in °C_
    entropy = coef["entropy_intercept"] + coef["entropy_slope"] * tc_growth

    # Calculate Arrhenius components
    fva = calculate_simple_arrhenius_factor(
        tk=tk_leaf, ha=coef["ha"], tk_ref=tk_ref, k_R=k_R
    )

    fvb = (1 + np.exp((tk_ref * entropy - coef["hd"]) / (k_R * tk_ref))) / (
        1 + np.exp((tk_leaf * entropy - coef["hd"]) / (k_R * tk_leaf))
    )

    return fva * fvb


def calc_ftemp_inst_rd(
    tc: NDArray[np.float64],
    tc_ref: float = PModelConst().tc_ref,
    coef: tuple[float, float] = PModelConst().heskel_rd,
) -> NDArray[np.float64]:
    r"""Calculate temperature scaling of dark respiration.

    Calculates the temperature-scaling factor for dark respiration at a given
    temperature (``tc``, :math:`T` in °C), relative to the standard reference
    temperature :math:`T_o`, given the parameterisation in :cite:t:`Heskel:2016fg`.

    .. math::

            fr = \exp( b (T_o - T) -  c ( T_o^2 - T^2 ))

    Args:
        tc: Temperature (:math:`T`, °C)
        tc_ref: standard reference temperature for photosynthetic processes
            (:math:`T_o`,°C)
        coef: A two tuple of floats providing the linear and quadratic coefficients
            (:math:`b` and :math:`c`)

    Examples:
        >>> # Relative instantaneous change in Rd going from 10 to 25 degrees
        >>> pmod_consts = PModelConst()
        >>> (
        ...     calc_ftemp_inst_rd(
        ...         tc=25, tc_ref=pmod_consts.tc_ref, coef=pmod_consts.heskel_rd
        ...     )
        ...     / calc_ftemp_inst_rd(
        ...         tc=10, tc_ref=pmod_consts.tc_ref, coef=pmod_consts.heskel_rd
        ...     )
        ...     - 1
        ... ).round(4)
        np.float64(2.5096)
    """

    return np.exp(coef[0] * (tc - tc_ref) - coef[1] * (tc**2 - tc_ref**2))


def calc_gammastar(
    tk: NDArray[np.float64],
    patm: NDArray[np.float64],
    tk_ref: float = PModelConst().tk_ref,
    k_Po: float = CoreConst().k_Po,
    k_R: float = CoreConst().k_R,
    coef: dict[str, float] = PModelConst().bernacchi_gs,
) -> NDArray[np.float64]:
    r"""Calculate the photorespiratory CO2 compensation point.

    Calculates the photorespiratory **CO2 compensation point** in absence of dark
    respiration (:math:`\Gamma^{*}`, :cite:alp:`Farquhar:1980ft`) as:

    .. math::

        \Gamma^{*} = \Gamma^{*}_{0} \cdot \frac{p}{p_0} \cdot f(T, H_a)

    where :math:`f(T, H_a)` modifies the activation energy to the the local temperature
    in Kelvin following the Arrhenius-type temperature response function (see
    :meth:`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`). By default,
    estimates of  :math:`\Gamma^{*}_{0}` and :math:`H_a` are taken from
    :cite:t:`Bernacchi:2001kg` (see
    :attr:`PModelConst.bernacchi_gs<pyrealm.constants.pmodel_const.PModelConst.bernacchi_gs>`)

    Args:
        tk: Temperature relevant for photosynthesis in Kelvin(:math:`T`, K)
        patm: Atmospheric pressure (:math:`p`, Pascals)
        tk_ref: The reference temperature of the coefficients in Kelvin.
        k_Po: The standard atmospheric pressure, defaulting to the value
            from :attr:`~pyrealm.constants.core_const.CoreConst.k_Po`.
        k_R: The universal gas constant, defaulting to the value from
            :attr:`~pyrealm.constants.core_const.CoreConst.k_R`.
        coef: A dictionary providing the enzyme kinetic coefficients for the reaction
            (``dha`` and ``gs25_0``).

    Returns:
        A float value or values for :math:`\Gamma^{*}` (in Pa)

    Examples:
        >>> # CO2 compensation point at 20 °C  (293.15 K) and standard presssure
        >>> calc_gammastar(np.array([293.15]), np.array([101325])).round(5)
        array([3.33925])
    """

    # check inputs, return shape not used
    _ = check_input_shapes(tk, patm)

    return (
        coef["gs25_0"]
        * patm
        / k_Po
        * calculate_simple_arrhenius_factor(
            tk=tk, tk_ref=tk_ref, ha=coef["dha"], k_R=k_R
        )
    )


def calc_ns_star(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate the relative viscosity of water.

    Calculates the relative viscosity of water (:math:`\eta^*`), given the standard
    temperature and pressure, using :func:`~pyrealm.core.water.calc_viscosity_h2o`
    (:math:`v(t,p)`) as:

    .. math::

        \eta^* = \frac{v(t,p)}{v(t_0,p_0)}

    Args:
        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        To: standard temperature (:math:`t0`, ``k_To``)
        Po: standard pressure (:math:`p_0`, ``k_Po``)

    Returns:
        A numeric value for :math:`\eta^*` (a unitless ratio)

    Examples:
        >>> # Relative viscosity at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa):
        >>> round(calc_ns_star(20, 101325), 5)
        np.float64(1.12536)
    """

    visc_env = calc_viscosity_h2o(tc, patm, core_const=core_const)
    visc_std = calc_viscosity_h2o(
        np.array(core_const.k_To) - np.array(core_const.k_CtoK),
        np.array(core_const.k_Po),
        core_const=core_const,
    )

    return visc_env / visc_std


def calc_kmm(
    tk: NDArray[np.float64],
    patm: NDArray[np.float64],
    tk_ref: float = PModelConst().tk_ref,
    k_co: float = CoreConst().k_co,
    k_R: float = CoreConst().k_R,
    coef: dict[str, float] = PModelConst().bernacchi_kmm,
) -> NDArray[np.float64]:
    r"""Calculate the Michaelis Menten coefficient of Rubisco-limited assimilation.

    Calculates the Michaelis Menten coefficient of Rubisco-limited assimilation
    (:math:`K`, :cite:alp:`Farquhar:1980ft`) as a function of temperature (:math:`T,
    Kelvin) and atmospheric pressure (:math:`p`, Pa) as:

      .. math:: K = K_c ( 1 + p_{\ce{O2}} / K_o),

    where, :math:`p_{\ce{O2}} = 0.209476 \cdot p` is the partial pressure of oxygen.
    :math:`f(T, H_a)` is the simple Arrhenius temperature response of activation
    energies (see :meth:`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`)
    used to correct Michalis constants at standard temperature for both :math:`\ce{CO2}`
    and :math:`\ce{O2}` to the local temperature. The default values for the enzyme
    coefficients are taken from Table 1 of :cite:t:`Bernacchi:2001kg` (see
    attr:`PModelConst.bernacchi_kmm<pyrealm.constants.pmodel_const.PModelConst.bernacchi_kmm>`)

      .. math::
        :nowrap:

        \[
            \begin{align*}
                K_c &= K_{c25} \cdot f(T, H_{kc})\\ K_o &= K_{o25} \cdot f(T, H_{ko})
            \end{align*}
        \]

    Args:
        tk: Temperature relevant for photosynthesis in Kelvin (:math:`T`, K)
        patm: Atmospheric pressure (:math:`p`, Pa)
        tk_ref: The reference temperature of the coefficients in Kelvin.
        k_co: The partial pressure of :math:`\ce{O2}` at standard pressure, defaulting
            to the values from  attr:`~pyrealm.constants.core_const.CoreConst.k_co`.
        k_R: The universal gas constant, defaulting to the value from
            attr:`~pyrealm.constants.core_const.CoreConst.k_R`.
        coef: A dictionary providing the enzyme kinetic coefficients for the reaction
            (``kc25``, ``ko25``, ``dhac``, ``dhao``).

    Returns:
        A numeric value for :math:`K` (in Pa)

    Examples:
        >>> # Michaelis-Menten coefficient at 20°C (293.15K) and standard pressure (Pa)
        >>> calc_kmm(np.array([293.15]), np.array([101325])).round(5)
        array([46.09928])
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tk, patm)

    kc = coef["kc25"] * calculate_simple_arrhenius_factor(
        tk=tk, tk_ref=tk_ref, ha=coef["dhac"], k_R=k_R
    )

    ko = coef["ko25"] * calculate_simple_arrhenius_factor(
        tk=tk, tk_ref=tk_ref, ha=coef["dhao"], k_R=k_R
    )

    # O2 partial pressure
    po = k_co * 1e-6 * patm

    return kc * (1.0 + po / ko)


def calc_soilmstress_stocker(
    soilm: NDArray[np.float64],
    meanalpha: NDArray[np.float64] = np.array(1.0),
    coef: dict[str, float] = PModelConst().soilmstress_stocker,
) -> NDArray[np.float64]:
    r"""Calculate Stocker's empirical soil moisture stress factor.

    This function calculates a penalty factor :math:`\beta(\theta)` for well-watered GPP
    estimates as an empirically derived stress factor :cite:p:`Stocker:2020dh`. The
    factor is calculated as a function of relative soil moisture (:math:`m_s`, fraction
    of field capacity) and average aridity, quantified by the local annual mean ratio of
    actual over potential evapotranspiration (:math:`\bar{\alpha}`).

    The value of :math:`\beta` is defined relative to two soil moisture thresholds
    (:math:`\theta_0, \theta^{*}`) as:

      .. math::
        :nowrap:

        \[
            \beta =
                \begin{cases}
                    q(m_s - \theta^{*})^2 + 1,  & \theta_0 < m_s <= \theta^{*} \\
                    1, &  \theta^{*} < m_s,
                \end{cases}
        \]

    where :math:`q` is an aridity sensitivity parameter setting the stress factor at
    :math:`\theta_0`:

    .. math:: q=(1 - (a + b \bar{\alpha}))/(\theta^{*} - \theta_{0})^2

    .. IMPORTANT::

        The default parameterisation of this water stress penalty (:math:`a=0`,
        :math:`b=0.7330`)  was estimated from empirical data using the **standard form
        of the PModel**  (:class:`~pyrealm.pmodel.pmodel.PModel` in ``pyrealm``).

        These parameters were then further calibrated against empirical data by tuning
        the quantum yield of photosynthesis. This tuning aimed to capture include
        incomplete leaf absorption in the realised value of :math:`\phi_0`, and
        :cite:t:`Stocker:2020dh` argue that, within their model representation,
        :math:`\phi_0` should be treated as a parameter representing canopy-scale
        effective quantum yield. To duplicate the model settings used with this soil
        moisture correction in Table 1 of :cite:t:`Stocker:2020dh`, use the following
        code:

        .. code-block:: python

            # The 'BRC' model setup
            PModel(
                ...
                method_kphio="temperature",
                method_arrhenius="simple",
                method_jmaxlim="wang17",
                method_optchi="prentice14",
                reference_kphio=0.081785,
            )

            # The 'ORG' model setup
            PModel(
                ...
                method_kphio="fixed",
                method_arrhenius="simple",
                method_jmaxlim="wang17",
                method_optchi="prentice14",
                reference_kphio=0.049977,
            )

    The :mod:`pyrealm.pmodel` module treats this factor purely as a penalty that can be
    applied after the estimation of GPP. In contrast, the `rpmodel` implementation uses
    the penalised GPP to back-calculate realistic :math:`J_{max}` and :math:`V_{cmax}`
    values that would give rise to the penalised GPP.

    Args:
        soilm: Relative soil moisture as a fraction of field capacity
            (unitless). Defaults to 1.0 (no soil moisture stress).
        meanalpha: Local annual mean ratio of actual over potential
            evapotranspiration, measure for average aridity. Defaults to 1.0.
        coef: A dictionary providing values of the coefficients ``theta0``,
            ``thetastar``, ``a`` and ``b``, defaulting to the values from 
            attr:`~pyrealm.constants.pmodel_const.PModelConst.soilmstress_stocker`.

    Returns:
        A numeric value or values for :math:`\beta`

    Examples:
        >>> # Proportion of well-watered GPP available at soil moisture of 0.2
        >>> calc_soilmstress_stocker(np.array([0.2])).round(5)
        array([0.88133])
    """

    # TODO - move soilm params into standalone param class for this function -
    #        keep the PModelConst cleaner?

    # Check inputs, return shape not used
    _ = check_input_shapes(soilm, meanalpha)

    # Calculate outstress
    y0 = coef["a"] + coef["b"] * meanalpha
    beta = (1.0 - y0) / (coef["theta0"] - coef["thetastar"]) ** 2
    outstress = 1.0 - beta * (soilm - coef["thetastar"]) ** 2

    # Filter wrt to thetastar
    outstress = np.where(soilm <= coef["thetastar"], outstress, 1.0)

    # Clip
    outstress = np.clip(outstress, 0.0, 1.0)

    return outstress


def calc_soilmstress_mengoli(
    soilm: NDArray[np.float64] = np.array(1.0),
    aridity_index: NDArray[np.float64] = np.array(1.0),
    coef: dict[str, float] = PModelConst().soilmstress_mengoli,
) -> NDArray[np.float64]:
    r"""Calculate the Mengoli et al. empirical soil moisture stress factor.

    This function calculates a penalty factor :math:`\beta(\theta)` for well-watered GPP
    estimates as an empirically derived stress factor :cite:p:`mengoli:2023a`. The
    factor is calculated from relative soil moisture as a fraction of field capacity
    (:math:`\theta`) and the long-run climatological aridity index for a site
    (:math:`\textrm{AI}`), calculated as (total PET)/(total precipitation) for a
    suitable time period.

    The factor is calculated using two constrained power functions for the maximal level
    (:math:`y`) of productivity and the threshold (:math:`psi`) at which that maximal
    level is reached.

      .. math::
        :nowrap:

        \[
            \begin{align*}
            y &= \min( a  \textrm{AI} ^ {b}, 1)\\
            \psi &= \min( a  \textrm{AI} ^ {b}, 1)\\
            \beta(\theta) &=
                \begin{cases}
                    y, & \theta \ge \psi \\
                    \dfrac{y}{\psi} \theta, & \theta \lt \psi \\
                \end{cases}\\
            \end{align*}
        \]

    .. IMPORTANT::

        The parameterisation of this water stress penalty was estimated from empirical
        data using the **subdaily form of the PModel**
        (:class:`~pyrealm.pmodel.pmodel.SubdailyPModel` in ``pyrealm``) with
        temperature dependence of the standard maximum quantum yield of photosynthesis
        (``phi_0``, :math:`\phi_0=1/8`). 
        
        There are minor differences in the implementation of the Subdaily P Model in
        ``pyrealm`` from that used to calibrate this function in
        :cite:p:`mengoli:2023a`. To get the closest match when applying this
        soil moisture correction, use the following settings:

        .. code-block:: python

            SubdailyPModel(
                ...
                method_kphio="temperature",
                method_arrhenius="simple",
                method_jmaxlim="wang17",
                method_optchi="prentice14",
                reference_kphio=1/8,
            )

        The ``reference_kphio=1/8`` value here is in fact the default value used when
        ``method_kphio="temperature"`` but is restated here for clarity.

    Args:
        soilm: Relative soil moisture (unitless).
        aridity_index: The climatological aridity index.
        coef: A dictionary providing values of the coefficients ``y_a``, ``y_b``,
            ``psi_a`` and ``psi_b``, defaulting to the values from 
            attr:`~pyrealm.constants.pmodel_const.PModelConst.soilmstress_mengoli`.

    Returns:
        A numeric value or values for :math:`f(\theta)`

    Examples:
        >>> import numpy as np
        >>> # Proportion of well-watered GPP available with soil moisture and aridity
        >>> # index values of 0.6
        >>> calc_soilmstress_mengoli(np.array([0.6]), np.array([0.6])).round(5)
        array([0.78023])
    """

    # TODO - move soilm params into standalone param class for this function -
    #        keep the PModelConst cleaner?

    # Check inputs, return shape not used
    _ = check_input_shapes(soilm, aridity_index)

    # Calculate maximal level and threshold
    y = np.minimum(
        coef["y_a"] * np.power(aridity_index, coef["y_b"]),
        1,
    )

    psi = np.minimum(
        coef["psi_a"] * np.power(aridity_index, coef["psi_b"]),
        1,
    )

    # Return factor
    return np.where(soilm >= psi, y, (y / psi) * soilm)


def calc_co2_to_ca(
    co2: NDArray[np.float64], patm: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Convert :math:`\ce{CO2}` ppm to Pa.

    Converts ambient :math:`\ce{CO2}` (:math:`c_a`) in part per million to Pascals,
    accounting for atmospheric pressure.

    Args:
        co2: atmospheric :math:`\ce{CO2}`, ppm
        patm (float): atmospheric pressure, Pa

    Returns:
        Ambient :math:`\ce{CO2}` in units of Pa

    Examples:
        >>> np.round(calc_co2_to_ca(413.03, 101325), 6)
        np.float64(41.850265)
    """

    return 1.0e-6 * co2 * patm  # Pa, atms. CO2
