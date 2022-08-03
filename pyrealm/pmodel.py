"""The pmodel Module.

This module provides the P Model class, implementing an optimality based model
of photosynthesis, along with a set of functions to convert environmental
forcings to key parameters used within the model.

The module also provides code for...  TODO: Update this.
"""

from typing import Optional, Union
from warnings import warn

import bottleneck as bn
import numpy as np

from pyrealm import ExperimentalFeatureWarning
from pyrealm.bounds_checker import bounds_checker
from pyrealm.param_classes import C3C4Params, IsotopesParams, PModelParams
from pyrealm.utilities import check_input_shapes, summarize_attrs

# TODO - Note that the typing currently does not enforce the dtype of ndarrays
#        but it looks like the upcoming np.typing module might do this.


def calc_density_h2o(
    tc: Union[float, np.ndarray],
    patm: Union[float, np.ndarray],
    pmodel_params: PModelParams = PModelParams(),
    safe: bool = True,
) -> Union[float, np.ndarray]:
    """Calculate water density.

    Calculates the **density of water** as a function of temperature and
    atmospheric pressure, using the Tumlirz Equation and coefficients calculated
    by :cite:`Fisher:1975tm`.

    Args:
        tc: air temperature, °C
        patm: atmospheric pressure, Pa
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.
        safe: Prevents the function from estimating density below -30°C, where the
            function behaves poorly

    Other Parameters:
        lambda_: polynomial coefficients of Tumlirz equation
            (`pmodel_params.fisher_dial_lambda`).
        Po: polynomial coefficients of Tumlirz equation
            (`pmodel_params.fisher_dial_Po`).
        Vinf: polynomial coefficients of Tumlirz equation
            (`pmodel_params.fisher_dial_Vinf`).

    Returns:
        Water density as a float in (g cm^-3)

    Examples:
        >>> round(calc_density_h2o(20, 101325), 3)
        998.206
    """

    # DESIGN NOTE:
    # It doesn't make sense to use this function for tc < 0, but in particular
    # the calculation shows wild numeric instability between -44 and -46 that
    # leads to numerous downstream issues - see the extreme values documentation.
    if safe and np.nanmin(tc) < -30:
        raise RuntimeError(
            "Water density calculations below about -30°C are "
            "unstable. See argument safe to calc_density_h2o"
        )

    # Check input shapes, shape not used
    _ = check_input_shapes(tc, patm)

    # Get powers of tc, including tc^0 = 1 for constant terms
    tc_pow = np.power.outer(tc, np.arange(0, 10))

    # Calculate lambda, (bar cm^3)/g:
    lambda_val = np.sum(
        np.array(pmodel_params.fisher_dial_lambda) * tc_pow[..., :5], axis=-1
    )

    # Calculate po, bar
    po_val = np.sum(np.array(pmodel_params.fisher_dial_Po) * tc_pow[..., :5], axis=-1)

    # Calculate vinf, cm^3/g
    vinf_val = np.sum(np.array(pmodel_params.fisher_dial_Vinf) * tc_pow, axis=-1)

    # Convert pressure to bars (1 bar <- 100000 Pa)
    pbar = 1e-5 * patm

    # Calculate the specific volume (cm^3 g^-1):
    spec_vol = vinf_val + lambda_val / (po_val + pbar)

    # Convert to density (g cm^-3) -> 1000 g/kg; 1000000 cm^3/m^3 -> kg/m^3:
    rho = 1e3 / spec_vol

    # CDLO: Method of Chen et al (1997) - I tested this to compare to the TerrA-P
    # code base but I don't think we need it. Preserving the code in case it is
    # needed in the future.
    #
    #  # Calculate density at 1 atm (kg/m^3):
    #  chen_po = np.array([0.99983952, 6.788260e-5 , -9.08659e-6 , 1.022130e-7 ,
    #                      -1.35439e-9 , 1.471150e-11, -1.11663e-13, 5.044070e-16,
    #                      -1.00659e-18])
    #  po = np.sum(np.array(chen_po) * tc_pow[..., :9], axis=-1)
    #
    #  # Calculate bulk modulus at 1 atm (bar):
    #  chen_ko = np.array([19652.17, 148.1830, -2.29995, 0.01281,
    #                      -4.91564e-5, 1.035530e-7])
    #  ko = np.sum(np.array(chen_ko) * tc_pow[..., :6], axis=-1)
    #
    #  # Calculate temperature dependent coefficients:
    #  chen_ca = np.array([3.26138, 5.223e-4, 1.324e-4, -7.655e-7, 8.584e-10])
    #  ca = np.sum(np.array(chen_ca) * tc_pow[..., :5], axis=-1)
    #
    #  chen_cb = np.array([7.2061e-5, -5.8948e-6, 8.69900e-8, -1.0100e-9, 4.3220e-12])
    #  cb = np.sum(np.array(chen_cb) * tc_pow[..., :5], axis=-1)
    #
    #  # Convert atmospheric pressure to bar (1 bar = 100000 Pa)
    #  pbar = 1.0e-5 * patm
    #
    #  rho = (ko + ca * pbar + cb * pbar ** 2.0)
    #  rho /= (ko + ca * pbar + cb * pbar ** 2.0 - pbar)
    #  rho *= 1e3 * po

    return rho


def calc_ftemp_arrh(
    tk: Union[float, np.ndarray],
    ha: float,
    pmodel_params: PModelParams = PModelParams(),
) -> Union[float, np.ndarray]:
    r"""Calculate enzyme kinetics scaling factor.

    Calculates the temperature-scaling factor :math:`f` for enzyme kinetics
    following an Arrhenius response for a given temperature (``tk``, :math:`T`)
    and activation energy (`ha`, :math:`H_a`).

    Arrhenius kinetics are described as:

    .. math::

        x(T)= exp(c - H_a / (T R))

    The temperature-correction function :math:`f(T, H_a)` is:

      .. math::
        :nowrap:

        \[
            \begin{align*}
                f &= \frac{x(T)}{x(T_0)} \\
                  &= exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)
                        \text{, or equivalently}\\
                  &= exp \left( \frac{ H_a}{R} \cdot
                        \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
            \end{align*}
        \]

    Args:
        tk: Temperature (in Kelvin)
        ha: Activation energy (in :math:`J \text{mol}^{-1}`)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        To: a standard reference temperature (:math:`T_0`, `pmodel_params.k_To`)
        R: the universal gas constant (:math:`R`, `pmodel_params.k_R`)

    Returns:
        A float value or values for :math:`f`

    Examples:
        >>> # Relative rate change from 25 to 10 degrees Celsius (percent change)
        >>> round((1.0-calc_ftemp_arrh( 283.15, 100000)) * 100, 4)
        88.1991
    """

    # Note that the following forms are equivalent:
    # exp( ha * (tk - 298.15) / (298.15 * kR * tk) )
    # exp( ha * (tc - 25.0)/(298.15 * kR * (tc + 273.15)) )
    # exp( (ha/kR) * (1/298.15 - 1/tk) )

    tkref = pmodel_params.k_To + pmodel_params.k_CtoK

    return np.exp(ha * (tk - tkref) / (tkref * pmodel_params.k_R * tk))


def calc_ftemp_inst_rd(
    tc: Union[float, np.ndarray], pmodel_params: PModelParams = PModelParams()
) -> Union[float, np.ndarray]:
    """Calculate temperature scaling of dark respiration.

    Calculates the temperature-scaling factor for dark respiration at a given
    temperature (``tc``, :math:`T` in °C), relative to the standard reference
    temperature :math:`T_o` (:cite:`Heskel:2016fg`).

    .. math::

            fr = exp( b (T_o - T) -  c ( T_o^2 - T^2 ))

    Args:
        tc: Temperature (degrees Celsius)

    Other Parameters:
        To: standard reference temperature (:math:`T_o`, `pmodel_params.k_To`)
        b: empirically derived global mean coefficient
            (:math:`b`, Table 1, ::cite:`Heskel:2016fg`)
        c: empirically derived global mean coefficient
            (:math:`c`, Table 1, ::cite:`Heskel:2016fg`)


    Returns:
        A float value for :math:`fr`

    Examples:
        >>> # Relative percentage instantaneous change in Rd going from 10 to 25 degrees
        >>> val = (calc_ftemp_inst_rd(25) / calc_ftemp_inst_rd(10) - 1) * 100
        >>> round(val, 4)
        250.9593
    """

    return np.exp(
        pmodel_params.heskel_b * (tc - pmodel_params.k_To)
        - pmodel_params.heskel_c * (tc**2 - pmodel_params.k_To**2)
    )


def calc_ftemp_inst_vcmax(
    tc: Union[float, np.ndarray], pmodel_params: PModelParams = PModelParams()
) -> Union[float, np.ndarray]:
    r"""Calculate temperature scaling of :math:`V_{cmax}`.

    This function calculates the temperature-scaling factor :math:`f` of
    the instantaneous temperature response of :math:`V_{cmax}`, given the
    temperature (:math:`T`) relative to the standard reference temperature
    (:math:`T_0`), following modified Arrhenius kinetics.

    .. math::

       V = f V_{ref}

    The value of :math:`f` is given by :cite:`Kattge:2007db` (Eqn 1) as:

    .. math::

        f = g(T, H_a) \cdot
                \frac{1 + exp( (T_0 \Delta S - H_d) / (T_0 R))}
                     {1 + exp( (T \Delta S - H_d) / (T R))}

    where :math:`g(T, H_a)` is a regular Arrhenius-type temperature response
    function (see :func:`calc_ftemp_arrh`). The term :math:`\Delta S` is the
    entropy factor, calculated as a linear function of :math:`T` in °C following
    :cite:`Kattge:2007db` (Table 3, Eqn 4):

    .. math::

        \Delta S = a + b T

    Args:
        tc:  temperature, or in general the temperature relevant for
            photosynthesis (°C)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        Ha: activation energy (:math:`H_a`, `pmodel_params.kattge_knorr_Ha`)
        Hd: deactivation energy (:math:`H_d`, `pmodel_params.kattge_knorr_Hd`)
        To: standard reference temperature expressed in Kelvin
            (`T_0`, `pmodel_params.k_To`)
        R: the universal gas constant (:math:`R`,`pmodel_params.k_R`)
        a: intercept of the entropy factor
            (:math:`a`, `pmodel_params.kattge_knorr_a_ent`)
        b: slope of the entropy factor (:math:`b`, `pmodel_params.kattge_knorr_b_ent`)

    Returns:
        A float value or values for :math:`f`

    Examples:
        >>> # Relative change in Vcmax going (instantaneously, i.e. not
        >>> # not acclimatedly) from 10 to 25 degrees (percent change):
        >>> val = ((calc_ftemp_inst_vcmax(25)/calc_ftemp_inst_vcmax(10)-1) * 100)
        >>> round(val, 4)
        283.1775
    """

    # Convert temperatures to Kelvin
    tkref = pmodel_params.k_To + pmodel_params.k_CtoK
    tk = tc + pmodel_params.k_CtoK

    # Calculate entropy following Kattge & Knorr (2007): slope and intercept
    # are defined using temperature in °C, not K!!! 'tcgrowth' corresponds
    # to 'tmean' in Nicks, 'tc25' is 'to' in Nick's
    dent = pmodel_params.kattge_knorr_a_ent + pmodel_params.kattge_knorr_b_ent * tc
    fva = calc_ftemp_arrh(tk, pmodel_params.kattge_knorr_Ha)
    fvb = (
        1
        + np.exp(
            (tkref * dent - pmodel_params.kattge_knorr_Hd) / (pmodel_params.k_R * tkref)
        )
    ) / (
        1
        + np.exp((tk * dent - pmodel_params.kattge_knorr_Hd) / (pmodel_params.k_R * tk))
    )

    return fva * fvb


def calc_ftemp_kphio(
    tc: Union[float, np.ndarray],
    c4: bool = False,
    pmodel_params: PModelParams = PModelParams(),
) -> Union[float, np.ndarray]:
    r"""Calculate temperature dependence of quantum yield efficiency.

    Calculates the **temperature dependence of the quantum yield efficiency**,
    as a quadratic function of temperature (:math:`T`). The values of the
    coefficients depend on whether C3 or C4 photosynthesis is being
    modelled

    .. math::

        \phi(T) = a + b T - c T^2

    The factor :math:`\phi(T)` is to be multiplied with leaf absorptance and the
    fraction of absorbed light that reaches photosystem II. In the P-model these
    additional factors are lumped into a single apparent quantum yield
    efficiency parameter (argument `kphio` to the class
    :class:`~pyrealm.pmodel.PModel`).

    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        c4: Boolean specifying whether fitted temperature response for C4 plants
            is used. Defaults to \code{FALSE}.
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        C3: the parameters (:math:`a,b,c`, `pmodel_params.kphio_C3`) are taken from the
            temperature dependence of the maximum quantum yield of photosystem
            II in light-adapted tobacco leaves determined by :cite:`Bernacchi:2003dc`.
        C4: the parameters (:math:`a,b,c`, `pmodel_params.kphio_C4`) are taken
            from :cite:`cai:2020a`.

    Returns:
        A float value or values for :math:`\phi(T)`

    Examples:
        >>> # Relative change in the quantum yield efficiency between 5 and 25
        >>> # degrees celsius (percent change):
        >>> val = (calc_ftemp_kphio(25.0) / calc_ftemp_kphio(5.0) - 1) * 100
        >>> round(val, 5)
        52.03969
        >>> # Relative change in the quantum yield efficiency between 5 and 25
        >>> # degrees celsius (percent change) for a C4 plant:
        >>> val = (calc_ftemp_kphio(25.0, c4=True) /
        ...        calc_ftemp_kphio(5.0, c4=True) - 1) * 100
        >>> round(val, 5)
        432.25806
    """

    if c4:
        coef = pmodel_params.kphio_C4
    else:
        coef = pmodel_params.kphio_C3

    ftemp = coef[0] + coef[1] * tc + coef[2] * tc**2
    ftemp = np.clip(ftemp, 0.0, None)

    return ftemp


def calc_gammastar(
    tc: Union[float, np.ndarray],
    patm: Union[float, np.ndarray],
    pmodel_params: PModelParams = PModelParams(),
) -> Union[float, np.ndarray]:
    r"""Calculate the photorespiratory CO2 compensation point.

    Calculates the photorespiratory **CO2 compensation point** in absence of
    dark respiration (:math:`\Gamma^{*}`, ::cite:`Farquhar:1980ft`) as:

    .. math::

        \Gamma^{*} = \Gamma^{*}_{0} \cdot \frac{p}{p_0} \cdot f(T, H_a)

    where :math:`f(T, H_a)` modifies the activation energy to the the local
    temperature following an Arrhenius-type temperature response function
    implemented in :func:`calc_ftemp_arrh`.

    Args:
        tc: Temperature relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pascals)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        To: the standard reference temperature (:math:`T_0` )
        Po: the standard pressure (:math:`p_0` )
        gs_0: the reference value of :math:`\Gamma^{*}` at standard temperature
            (:math:`T_0`) and pressure (:math:`P_0`)  (:math:`\Gamma^{*}_{0}`,
            ::cite:`Bernacchi:2001kg`, `pmodel_params.bernacchi_gs25_0`)
        ha: the activation energy (:math:`\Delta H_a`, ::cite:`Bernacchi:2001kg`,
            `pmodel_params.bernacchi_dha`)

    Returns:
        A float value or values for :math:`\Gamma^{*}` (in Pa)

    Examples:
        >>> # CO2 compensation point at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa) >>> round(calc_gammastar(20, 101325), 5)
        3.33925
    """

    # check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    return (
        pmodel_params.bernacchi_gs25_0
        * patm
        / pmodel_params.k_Po
        * calc_ftemp_arrh((tc + pmodel_params.k_CtoK), ha=pmodel_params.bernacchi_dha)
    )


def calc_ns_star(
    tc: Union[float, np.ndarray],
    patm: Union[float, np.ndarray],
    pmodel_params: PModelParams = PModelParams(),
) -> Union[float, np.ndarray]:
    r"""Calculate the relative viscosity of water.

    Calculates the relative viscosity of water (:math:`\eta^*`), given the
    standard temperature and pressure, using :func:`~pyrealm.pmodel.calc_viscosity_h20`
    (:math:`v(t,p)`) as:

    .. math::

        \eta^* = \frac{v(t,p)}{v(t_0,p_0)}

    Args:
        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        To: standard temperature (:math:`t0`, `pmodel_params.k_To`)
        Po: standard pressure (:math:`p_0`, `pmodel_params.k_Po`)

    Returns:
        A numeric value for :math:`\eta^*` (a unitless ratio)

    Examples:
        >>> # Relative viscosity at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa):
        >>> round(calc_ns_star(20, 101325), 5)
        1.12536
    """

    visc_env = calc_viscosity_h2o(tc, patm, pmodel_params=pmodel_params)
    visc_std = calc_viscosity_h2o(
        pmodel_params.k_To, pmodel_params.k_Po, pmodel_params=pmodel_params
    )

    return visc_env / visc_std


def calc_kmm(
    tc: Union[float, np.ndarray],
    patm: Union[float, np.ndarray],
    pmodel_params: PModelParams = PModelParams(),
) -> Union[float, np.ndarray]:
    r"""Calculate the Michaelis Menten coefficient of Rubisco-limited assimilation.

    Calculates the **Michaelis Menten coefficient of Rubisco-limited
    assimilation** (:math:`K`, ::cite:`Farquhar:1980ft`) as a function of
    temperature (:math:`T`) and atmospheric pressure (:math:`p`) as:

      .. math:: K = K_c ( 1 + p_{\ce{O2}} / K_o),

    where, :math:`p_{\ce{O2}} = 0.209476 \cdot p` is the partial pressure of
    oxygen. :math:`f(T, H_a)` is an Arrhenius-type temperature response of
    activation energies (:func:`calc_ftemp_arrh`) used to correct
    Michalis constants at standard temperature for both :math:`\ce{CO2}` and
    :math:`\ce{O2}` to the local temperature (Table 1, ::cite:`Bernacchi:2001kg`):

      .. math::
        :nowrap:

        \[
            \begin{align*}
                K_c &= K_{c25} \cdot f(T, H_{kc})\\
                K_o &= K_{o25} \cdot f(T, H_{ko})
            \end{align*}
        \]

    .. TODO - why this height? Inconsistent with calc_gammastar which uses P_0
              for the same conversion for a value in the same table.

    Args:
        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        hac: activation energy for :math:`\ce{CO2}`
            (:math:`H_{kc}`, `pmodel_params.bernacchi_dhac`)
        hao:  activation energy for :math:`\ce{O2}`
            (:math:`\Delta H_{ko}`, `pmodel_params.bernacchi_dhao`)
        kc25: Michelis constant for :math:`\ce{CO2}` at standard temperature
            (:math:`K_{c25}`, `pmodel_params.bernacchi_kc25`)
        ko25: Michelis constant for :math:`\ce{O2}` at standard temperature
            (:math:`K_{o25}`, `pmodel_params.bernacchi_ko25`)

    Returns:
        A numeric value for :math:`K` (in Pa)

    Examples:
        >>> # Michaelis-Menten coefficient at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa):
        >>> round(calc_kmm(20, 101325), 5)
        46.09928
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    # conversion to Kelvin
    tk = tc + pmodel_params.k_CtoK

    kc = pmodel_params.bernacchi_kc25 * calc_ftemp_arrh(
        tk, ha=pmodel_params.bernacchi_dhac
    )
    ko = pmodel_params.bernacchi_ko25 * calc_ftemp_arrh(
        tk, ha=pmodel_params.bernacchi_dhao
    )

    # O2 partial pressure
    po = pmodel_params.k_co * 1e-6 * patm

    return kc * (1.0 + po / ko)


def calc_kp_c4(
    tc: Union[float, np.ndarray],
    patm: Union[float, np.ndarray],
    pmodel_params: PModelParams = PModelParams(),
) -> Union[float, np.ndarray]:
    r"""Calculate the Michaelis Menten coefficient of PEPc.

    Calculates the Michaelis Menten coefficient of phosphoenolpyruvate
    carboxylase (PEPc) (:math:`K`, :cite:`boyd:2015a`) as a function of
    temperature (:math:`T`) and atmospheric pressure (:math:`p`) as:

    Args:
        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        hac: activation energy for :math:`\ce{CO2}` (:math:`H_{kc}`,
             `pmodel_params.boyd_dhac_c4`)
        kc25: Michelis constant for :math:`\ce{CO2}` at standard temperature
            (:math:`K_{c25}`, `pmodel_params.boyd_kp25_c4`)

    Returns:
        A numeric value for :math:`K` (in Pa)

    Examples:
        >>> # Michaelis-Menten coefficient at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa):
        >>> round(calc_kp_c4(20, 101325), 5)
        9.26352
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    # conversion to Kelvin
    tk = tc + pmodel_params.k_CtoK
    return pmodel_params.boyd_kp25_c4 * calc_ftemp_arrh(
        tk, ha=pmodel_params.boyd_dhac_c4
    )


def calc_soilmstress(
    soilm: Union[float, np.ndarray],
    meanalpha: Union[float, np.ndarray] = 1.0,
    pmodel_params: PModelParams = PModelParams(),
) -> Union[float, np.ndarray]:
    r"""Calculate Stocker's empirical soil moisture stress factor.

    Calculates an **empirical soil moisture stress factor**  (:math:`\beta`,
    ::cite:`Stocker:2020dh`) as a function of relative soil moisture
    (:math:`m_s`, fraction of field capacity) and average aridity, quantified by
    the local annual mean ratio of actual over potential evapotranspiration
    (:math:`\bar{\alpha}`).

    The value of :math:`\beta` is defined relative to two soil moisture
    thresholds (:math:`\theta_0, \theta^{*}`) as:

      .. math::
        :nowrap:

        \[
            \beta =
                \begin{cases}
                    0
                    q(m_s - \theta^{*})^2 + 1,  & \theta_0 < m_s <= \theta^{*} \\
                    1, &  \theta^{*} < m_s,
                \end{cases}
        \]

    where :math:`q` is an aridity sensitivity parameter setting the stress
    factor at :math:`\theta_0`:

    .. math:: q=(1 - (a + b \bar{\alpha}))/(\theta^{*} - \theta_{0})^2

    Default parameters of :math:`a=0` and :math:`b=0.7330` are as described in
    Table 1 of :cite:`Stocker:2020dh` specifically for the 'FULL' use case, with
    ``method_jmaxlim="wang17"``, ``do_ftemp_kphio=TRUE``.

    Args:
        soilm: Relative soil moisture as a fraction of field capacity
            (unitless). Defaults to 1.0 (no soil moisture stress).
        meanalpha: Local annual mean ratio of actual over potential
            evapotranspiration, measure for average aridity. Defaults to 1.0.
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        theta0: lower bound of soil moisture
            (:math:`\theta_0`, `pmodel_params.soilmstress_theta0`).
        thetastar: upper bound of soil moisture
            (:math:`\theta^{*}`, `pmodel_params.soilmstress_thetastar`).
        a: aridity parameter (:math:`a`, `pmodel_params.soilmstress_a`).
        b: aridity parameter (:math:`b`, `pmodel_params.soilmstress_b`).

    Returns:
        A numeric value or values for :math:`\beta`

    Examples:
        >>> # Relative reduction (%) in GPP due to soil moisture stress at
        >>> # relative soil water content ('soilm') of 0.2:
        >>> round((calc_soilmstress(0.2) - 1) * 100, 5)
        -11.86667
    """

    # TODO - presumably this should also have beta(theta) = 0 when m_s <=
    #        theta_0. Actually, no - these limits aren't correct. This is only
    #        true when meanalpha=0, otherwise beta > 0 when m_s < theta_0.
    # TODO - move soilm params into standalone param class for this function -
    #        keep the PModelParams cleaner?

    # Check inputs, return shape not used
    _ = check_input_shapes(soilm, meanalpha)

    # Calculate outstress
    y0 = pmodel_params.soilmstress_a + pmodel_params.soilmstress_b * meanalpha
    beta = (1.0 - y0) / (
        pmodel_params.soilmstress_theta0 - pmodel_params.soilmstress_thetastar
    ) ** 2
    outstress = 1.0 - beta * (soilm - pmodel_params.soilmstress_thetastar) ** 2

    # Filter wrt to thetastar
    outstress = np.where(soilm <= pmodel_params.soilmstress_thetastar, outstress, 1.0)

    # Clip
    outstress = np.clip(outstress, 0.0, 1.0)

    return outstress


def calc_viscosity_h2o(
    tc: Union[float, np.ndarray],
    patm: Union[float, np.ndarray],
    pmodel_params: PModelParams = PModelParams(),
    simple: bool = False,
) -> Union[float, np.ndarray]:
    r"""Calculate the viscosity of water.

    Calculates the viscosity of water (:math:`\eta`) as a function of
    temperature and atmospheric pressure (::cite:`Huber:2009fy`).

    Args:
        tc: air temperature (°C)
        patm: atmospheric pressure (Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.
        simple: Use the simple formulation.

    Returns:
        A float giving the viscosity of water (mu, Pa s)

    Examples:
        >>> # Density of water at 20 degrees C and standard atmospheric pressure:
        >>> round(calc_viscosity_h2o(20, 101325), 7)
        0.0010016
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    if simple or pmodel_params.simple_viscosity:
        # The reference for this is unknown, but is used in some implementations
        # so is included here to allow intercomparison.
        return np.exp(-3.719 + 580 / ((tc + 273) - 138))

    # Get the density of water, kg/m^3
    rho = calc_density_h2o(tc, patm, pmodel_params=pmodel_params)

    # Calculate dimensionless parameters:
    tbar = (tc + pmodel_params.k_CtoK) / pmodel_params.huber_tk_ast
    rbar = rho / pmodel_params.huber_rho_ast

    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    tbar_pow = np.power.outer(tbar, np.arange(0, 4))
    mu0 = (1e2 * np.sqrt(tbar)) / np.sum(
        np.array(pmodel_params.huber_H_i) / tbar_pow, axis=-1
    )

    # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
    h_array = np.array(pmodel_params.huber_H_ij)
    ctbar = (1.0 / tbar) - 1.0
    row_j, _ = np.indices(h_array.shape)
    mu1 = h_array * np.power.outer(rbar - 1.0, row_j)
    mu1 = np.power.outer(ctbar, np.arange(0, 6)) * np.sum(mu1, axis=(-2))
    mu1 = np.exp(rbar * mu1.sum(axis=-1))

    # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
    mu_bar = mu0 * mu1

    # Calculate mu (Eq. 1, Huber et al., 2009)
    return mu_bar * pmodel_params.huber_mu_ast  # Pa s


def calc_patm(
    elv: Union[float, np.ndarray], pmodel_params: PModelParams = PModelParams()
) -> Union[float, np.ndarray]:
    r"""Calculate atmospheric pressure from elevation.

    Calculates atmospheric pressure as a function of elevation with reference to
    the standard atmosphere.  The elevation-dependence of atmospheric pressure
    is computed by assuming a linear decrease in temperature with elevation and
    a mean adiabatic lapse rate (Eqn 3, ::cite:`BerberanSantos:2009bk`):

    .. math::

        p(z) = p_0 ( 1 - L z / K_0) ^{ G M / (R L) },

    Args:
        elv: Elevation above sea-level (:math:`z`, metres above sea level.)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:
        G: gravity constant (:math:`g`, `pmodel_params.k_G`)
        Po: standard atmospheric pressure at sea level
            (:math:`p_0`, `pmodel_params.k_Po`)
        L: adiabatic temperature lapse rate (:math:`L}`, `pmodel_params.k_L`),
        M: molecular weight for dry air (:math:`M`, `pmodel_params.k_Ma`),
        R: universal gas constant (:math:`R`, `pmodel_params.k_R`)
        Ko: reference temperature in Kelvin (:math:`K_0`, `pmodel_params.k_To`).

    Returns:
        A numeric value for :math:`p` in Pascals.

    Examples:
        >>> # Standard atmospheric pressure, in Pa, corrected for 1000 m.a.s.l.
        >>> round(calc_patm(1000), 2)
        90241.54
    """

    # Convert elevation to pressure, Pa. This equation uses the base temperature
    # in Kelvins, while other functions use this constant in the PARAM units of
    # °C.

    kto = pmodel_params.k_To + pmodel_params.k_CtoK

    return pmodel_params.k_Po * (1.0 - pmodel_params.k_L * elv / kto) ** (
        pmodel_params.k_G * pmodel_params.k_Ma / (pmodel_params.k_R * pmodel_params.k_L)
    )


def calc_co2_to_ca(
    co2: Union[float, np.ndarray],
    patm: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Convert :math:`\ce{CO2}` ppm to Pa.

    Converts ambient :math:`\ce{CO2}` (:math:`c_a`) in part per million to
    Pascals, accounting for atmospheric pressure.

    Args
        co2 (float): atmospheric :math:`\ce{CO2}`, ppm
        patm (float): atmospheric pressure, Pa

    Returns:
        Ambient :math:`\ce{CO2}` in units of Pa

    Examples:
        >>> np.round(calc_co2_to_ca(413.03, 101325), 6)
        41.850265
    """

    return 1.0e-6 * co2 * patm  # Pa, atms. CO2


# Design notes on PModel (0.3.1 -> 0.4.0)
# The PModel until 0.3.1 was a single class taking tc etc. as inputs. However
# a common use case would be to look at how the PModel predictions change with
# different options. I (DO) did consider retaining the single class and having
# PModel __init__ create the environment and then have PModel.fit(), but
# having two classes seemed to better separate the physiological model (PModel
# class and attributes) from the environment the model is being fitted to and
# also creates _separate_ PModel objects.

# Design notes on PModel (0.4.0 -> 0.5.0)
# In separating PPFD and FAPAR into a second step, I had created the IabsScaled
# class to store variables that scale linearly with Iabs. That class held and
# exposed scaled and unscaled versions of several parameteres. For a start, that
# was a bad class name since the values could be unscaled, but also most of the
# unscaled versions are never really used. Really (hat tip, Keith Bloomfield),
# there are two meaningful _efficiency_ variables (LUE, IWUE) and then a set of
# productivity variables. The new structure reflects that, removing the
# un-needed unscaled variables and simplifying the structure.


class PModelEnvironment:
    r"""Create a PModelEnvironment instance.

    This class takes the four key environmental inputs to the P Model and
    calculates four photosynthetic variables for those environmental
    conditions:

    * the photorespiratory CO2 compensation point (:math:`\Gamma^{*}`,
      :func:`~pyrealm.pmodel.calc_gammastar`),
    * the relative viscosity of water (:math:`\eta^*`,
      :func:`~pyrealm.pmodel.calc_ns_star`),
    * the ambient partial pressure of :math:`\ce{CO2}` (:math:`c_a`,
      :func:`~pyrealm.pmodel.calc_c02_to_ca`) and
    * the Michaelis Menten coefficient of Rubisco-limited assimilation
      (:math:`K`, :func:`~pyrealm.pmodel.calc_kmm`).

    These variables can then be used to fit P models using different
    configurations. Note that the underlying parameters of the P model
    (:class:`~pyrealm.param_classes.PModelParams`) are set when creating
    an instance of this class.

    In addition to the four key variables above, the PModelEnvironment class
    is used to provide additional variables used by some methods.

    * the volumetric soil moisture content, required to calculate optimal
      :math:`\chi` in :meth:`~pyrealm.pmodel.CalcOptimalChi.laverge2020`.

    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        vpd: Vapour pressure deficit (Pa)
        co2: Atmospheric CO2 concentration (ppm)
        patm: Atmospheric pressure (Pa)
        theta: Volumetric soil moisture (m3/m3)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.
    """

    def __init__(
        self,
        tc: Union[float, np.ndarray],
        vpd: Union[float, np.ndarray],
        co2: Union[float, np.ndarray],
        patm: Union[float, np.ndarray],
        theta: Optional[Union[float, np.ndarray]] = None,
        pmodel_params: PModelParams = PModelParams(),
    ):

        self.shape = check_input_shapes(tc, vpd, co2, patm)

        # Validate and store the forcing variables
        self.tc = bounds_checker(tc, -25, 80, "[]", "tc", "°C")
        self.vpd = bounds_checker(vpd, 0, 10000, "[]", "vpd", "Pa")
        self.co2 = bounds_checker(co2, 0, 1000, "[]", "co2", "ppm")
        self.patm = bounds_checker(patm, 30000, 110000, "[]", "patm", "Pa")

        # Guard against calc_density issues
        if np.nanmin(self.tc) < -25:
            raise ValueError(
                "Cannot calculate P Model predictions for values below"
                " -25°C. See calc_density_h2o."
            )

        # Guard against negative VPD issues
        if np.nanmin(self.vpd) < 0:
            raise ValueError(
                "Negative VPD values will lead to missing data - clip to "
                "zero or explicitly set to np.nan"
            )

        # ambient CO2 partial pressure (Pa)
        self.ca = calc_co2_to_ca(self.co2, self.patm)

        # photorespiratory compensation point - Gamma-star (Pa)
        self.gammastar = calc_gammastar(tc, patm, pmodel_params=pmodel_params)

        # Michaelis-Menten coef. (Pa)
        self.kmm = calc_kmm(tc, patm, pmodel_params=pmodel_params)

        # # Michaelis-Menten coef. C4 plants (Pa) NOT CHECKED. Need to think
        # # about how many optional variables stack up in PModelEnvironment
        # # and this is only required by C4 optimal chi Scott and Smith, which
        # # has not yet been implemented.
        # self.kp_c4 = calc_kp_c4(tc, patm, pmodel_params=pmodel_params)

        # viscosity correction factor relative to standards
        self.ns_star = calc_ns_star(tc, patm, pmodel_params=pmodel_params)  # (unitless)

        # Optional variables
        if theta is None:
            self.theta = None
        else:
            # Is the input congruent with the other variables and in bounds.
            _ = check_input_shapes(tc, theta)
            self.theta = bounds_checker(theta, 0, 0.8, "[]", "theta", "m3/m3")

        # Store parameters
        self.pmodel_params = pmodel_params

    def __repr__(self):

        # DESIGN NOTE: This is deliberately extremely terse. It could contain
        # a bunch of info on the environment but that would be quite spammy
        # on screen. Having a specific summary method that provides that info
        # is more user friendly.

        return f"PModelEnvironment(shape={self.shape})"

    def summarize(self, dp: int = 2) -> None:
        """Print PModelEnvironment summary.

        Prints a summary of the input and photosynthetic attributes in a
        instance of a PModelEnvironment including the mean, range and number
        of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [
            ("tc", "°C"),
            ("vpd", "Pa"),
            ("co2", "ppm"),
            ("patm", "Pa"),
            ("ca", "Pa"),
            ("gammastar", "Pa"),
            ("kmm", "Pa"),
            ("ns_star", "-"),
        ]
        summarize_attrs(self, attrs, dp=dp)


class PModel:
    r"""Fit the P Model.

    This class fits the P Model to a given set of environmental and
    photosynthetic parameters. The calculated attributes of the class are
    described below. An extended description with typical use cases is given in
    :any:`pmodel_overview` but the basic flow of the model is:

    1. Estimate :math:`\ce{CO2}` limitation factors and optimal internal to
       ambient :math:`\ce{CO2}` partial pressure ratios (:math:`\chi`), using
       :class:`~pyrealm.pmodel.CalcOptimalChi`.
    2. Estimate limitation factors to :math:`V_{cmax}` and :math:`J_{max}`
       using :class:`~pyrealm.pmodel.JmaxLimitation`.
    3. Optionally, estimate productivity measures including GPP by supplying
       FAPAR and PPFD using the
       :meth:`~pyrealm.pmodel.PModel.estimate_productivity` method.

    The model predictions from step 1 and 2 are then:

    * Intrinsic water use efficiency (iWUE,
      :math:`\mu\mathrm{mol}\;\mathrm{mol}^{-1}`), calculated as :math:`( 5/8 *
      (c_a - c_i)) / P`, where `c_a` and `c_i` are measured in Pa and :math:`P`
      is atmospheric pressure in megapascals. This is equivalent to :math:`(c_a
      - c_i)/1.6` when `c_a` and `c_i` are expressed as parts per million.

    * The light use efficienciy (LUE, gC mol-1), calculated as:

        .. math::

            \text{LUE} = \phi_0 \cdot m_j \cdot f_v \cdot M_C \cdot \beta(\theta),

      where :math:`f_v` is a limitation factor defined in
      :class:`~pyrealm.pmodel.JmaxLimitation`, :math:`M_C` is the molar mass of
      carbon and :math:`\beta(\theta)` is an empirical soil moisture factor
      (see :func:`~pyrealm.pmodel.calc_soilmstress`,  :cite:`Stocker:2020dh`).

    After running :meth:`~pyrealm.pmodel.PModel.estimate_productivity`, the following
    predictions are also populated:

    * Gross primary productivity, calculated as
        :math:`\text{GPP} = \text{LUE} \cdot I_{abs}`, where :math:`I_{abs}` is
        the absorbed photosynthetic radiation

    * The maximum rate of Rubisco regeneration at the growth temperature
        (:math:`J_{max}`)

    * The maximum carboxylation capacity (mol C m-2) at the growth temperature
        (:math:`V_{cmax}`).

    These two predictions are calculated as follows:

        .. math::
            :nowrap:

            \[
                \begin{align*}
                V_{cmax} &= \phi_{0} I_{abs} \frac{m}{m_c} f_{v} \\
                J_{max} &= 4 \phi_{0} I_{abs} f_{j} \\
                \end{align*}
            \]

    where  :math:`f_v, f_j` are limitation terms described in
    :class:`~pyrealm.pmodel.JmaxLimitation`

    * The maximum carboxylation capacity (mol C m-2) normalised to the standard
      temperature as: :math:`V_{cmax25} = V_{cmax}  / fv(t)`, where :math:`fv(t)` is
      the instantaneous temperature response of :math:`V_{cmax}` implemented in
      :func:`~pyrealm.pmodel.calc_ftemp_inst_vcmax`

    * Dark respiration, calculated as:

        .. math::

            R_d = b_0 \frac{fr(t)}{fv(t)} V_{cmax}

      following :cite:`Atkin:2015hk`, where :math:`fr(t)` is the instantaneous
      temperature response of dark respiration implemented in
      :func:`~pyrealm.pmodel.calc_ftemp_inst_rd`, and :math:`b_0` is set
      in :attr:`~pyrealm.pmodel_params.atkin_rd_to_vcmax`.

    * Stomatal conductance (:math:`g_s`), calculated as:

        .. math::

            g_s = \frac{LUE}{M_C}\frac{1}{c_a - c_i}

      When C4 photosynthesis is being used, the true partial pressure of CO2
      in the substomatal cavities (:math:`c_i`) is used following the
      calculation of :math:`\chi` using
      :attr:`~pyrealm.param_classes.PModelParams.beta_cost_ratio_c4`

    Soil moisture effects:
        The `soilmstress`, `rootzonestress` arguments and the `lavergne20_c3` and
        `lavergne20_c4` all implement different approaches to soil moisture effects on
        photosynthesis and are incompatible.

    Parameters:
        env: An instance of :class:`~pyrealm.pmodel.PModelEnvironment`.
        kphio: (Optional) The quantum yield efficiency of photosynthesis
            (:math:`\phi_0`, unitless). Note that :math:`\phi_0` is
            sometimes used to refer to the quantum yield of electron transfer,
            which is exactly four times larger, so check definitions here.
        rootzonestress: (Optional, default=None) An experimental option
            for providing a root zone water stress penalty to the :math:`beta` parameter
            in :class:`~pyrealm.pmodel.CalcOptimalChi`.
        soilmstress: (Optional, default=None) A soil moisture stress factor
            calculated using :func:`~pyrealm.pmodel.calc_soilmstress`.
        method_optchi: (Optional, default=`prentice14`) Selects the method to be
            used for calculating optimal :math:`chi`. The choice of method also
            sets the choice of  C3 or C4 photosynthetic pathway (see
            :class:`~pyrealm.pmodel.CalcOptimalChi`).
        method_jmaxlim: (Optional, default=`wang17`) Method to use for
            :math:`J_{max}` limitation
        do_ftemp_kphio: (Optional, default=True) Include the temperature-
            dependence of quantum yield efficiency (see
            :func:`~pyrealm.pmodel.calc_ftemp_kphio`).

    Attributes:
        env: The photosynthetic environment to which the model is fitted
            (:class:`~pyrealm.pmodel.PModelEnvironment`)
        pmodel_params: The parameters used in the underlying calculations
            (:class:`~pyrealm.param_classes.PModelParams`)
        optchi: Details of the optimal chi calculation
            (:class:`~pyrealm.pmodel.CalcOptimalChi`)
        init_kphio: The initial value of :math:`\phi_0`.
        kphio: The value of :math:`\phi_0` used in calculations with
            any temperature correction applied.
        iwue: Intrinsic water use efficiency (iWUE, µmol mol-1)
        lue: Light use efficiency (LUE, g C mol-1)

    After :meth:`~pyrealm.pmodel.estimate_productivity` has been run,
    the following attributes are also populated. See the documentation
    for :meth:`~pyrealm.pmodel.estimate_productivity` for details.

    Attributes:
        gpp: Gross primary productivity (µg C m-2 s-1)
        rd: Dark respiration (µmol m-2 s-1)
        vcmax: Maximum rate of carboxylation (µmol m-2 s-1)
        vcmax25: Maximum rate of carboxylation at standard temperature (µmol m-2 s-1)
        jmax: Maximum rate of electron transport (µmol m-2 s-1)
        gs: Stomatal conductance (µmol m-2 s-1)

    Examples:
        >>> env = PModelEnvironment(tc=20, vpd=1000, co2=400, patm=101325.0)
        >>> mod_c3 = PModel(env)
        >>> # Key variables from pmodel
        >>> np.round(mod_c3.optchi.ci, 5)
        28.14209
        >>> np.round(mod_c3.optchi.chi, 5)
        0.69435
        >>> mod_c3.estimate_productivity(fapar=1, ppfd=300)
        >>> np.round(mod_c3.gpp, 5)
        76.42545
        >>> mod_c4 = PModel(env, method_optchi='c4', method_jmaxlim='none')
        >>> # Key variables from PModel
        >>> np.round(mod_c4.optchi.ci, 5)
        18.22528
        >>> np.round(mod_c4.optchi.chi, 5)
        0.44967
        >>> mod_c4.estimate_productivity(fapar=1, ppfd=300)
        >>> np.round(mod_c4.gpp, 5)
        103.25886
    """

    def __init__(
        self,
        env: PModelEnvironment,
        rootzonestress: Optional[Union[float, np.ndarray]] = None,
        soilmstress: Optional[Union[float, np.ndarray]] = None,
        kphio: Optional[float] = None,
        do_ftemp_kphio: bool = True,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
    ):

        # Check possible array inputs against the photosynthetic environment
        self.shape = check_input_shapes(env.gammastar, soilmstress, rootzonestress)

        # Store a reference to the photosynthetic environment and a direct
        # reference to the parameterisation
        self.env = env
        self.pmodel_params = env.pmodel_params

        # ---------------------------------------------
        # Soil moisture and root zone stress handling
        # ---------------------------------------------

        if (
            (soilmstress is not None)
            + (rootzonestress is not None)
            + (method_optchi in ("lavergne20_c3", "lavergne20_c4"))
        ) > 1:
            raise AttributeError(
                "Soilmstress, rootzonestress and the lavergne20 method_optchi options "
                "are parallel approaches to soil moisture effects and cannot be "
                "combined."
            )

        if soilmstress is None:
            self.soilmstress = 1.0
            self.do_soilmstress = False
        else:
            self.soilmstress = soilmstress
            self.do_soilmstress = True

        if rootzonestress is None:
            self.do_rootzonestress = False
        else:
            warn(
                "The rootzonestress option is an experimental penalty factor to beta",
                ExperimentalFeatureWarning,
            )
            self.do_rootzonestress = True

        # kphio defaults:
        self.do_ftemp_kphio = do_ftemp_kphio
        if kphio is None:
            if not self.do_ftemp_kphio:
                self.init_kphio = 0.049977
            elif self.do_soilmstress:
                self.init_kphio = 0.087182
            else:
                self.init_kphio = 0.081785
        else:
            self.init_kphio = kphio

        # Check method_optchi and set c3/c4
        self.c4 = CalcOptimalChi._method_lookup(method_optchi)
        self.method_optchi = method_optchi

        # -----------------------------------------------------------------------
        # Temperature dependence of quantum yield efficiency
        # -----------------------------------------------------------------------
        if self.do_ftemp_kphio:
            ftemp_kphio = calc_ftemp_kphio(
                env.tc, self.c4, pmodel_params=env.pmodel_params
            )
            self.kphio = self.init_kphio * ftemp_kphio
        else:
            self.kphio = self.init_kphio

        # -----------------------------------------------------------------------
        # Optimal ci
        # The heart of the P-model: calculate ci:ca ratio (chi) and additional terms
        # -----------------------------------------------------------------------
        self.optchi = CalcOptimalChi(
            env=env,
            method=method_optchi,
            rootzonestress=rootzonestress,
            pmodel_params=env.pmodel_params,
        )

        # -----------------------------------------------------------------------
        # Calculation of Jmax limitation terms
        # -----------------------------------------------------------------------
        self.method_jmaxlim = method_jmaxlim

        self.jmaxlim = JmaxLimitation(
            self.optchi, method=self.method_jmaxlim, pmodel_params=env.pmodel_params
        )

        # -----------------------------------------------------------------------
        # Store the two efficiency predictions
        # -----------------------------------------------------------------------

        # Intrinsic water use efficiency (in µmol mol-1). The rpmodel reports this
        # in Pascals, but more commonly reported in µmol mol-1. The standard equation
        # (ca - ci) / 1.6 expects inputs in ppm, so the pascal versions are back
        # converted here.
        self.iwue = (5 / 8 * (env.ca - self.optchi.ci)) / (1e-6 * self.env.patm)

        # The basic calculation of LUE = phi0 * M_c * m but here we implement
        # two penalty terms for jmax limitation and Stocker beta soil moisture
        # stress
        # Note: the rpmodel implementation also estimates soilmstress effects on
        #       jmax and vcmax but pyrealm.pmodel only applies the stress factor
        #       to LUE and hence GPP
        self.lue = (
            self.kphio
            * self.optchi.mj
            * self.jmaxlim.f_v
            * self.pmodel_params.k_c_molmass
            * self.soilmstress
        )

        # -----------------------------------------------------------------------
        # Define attributes populated by estimate_productivity method
        # -----------------------------------------------------------------------
        self._vcmax = None
        self._vcmax25 = None
        self._rd = None
        self._jmax = None
        self._gpp = None
        self._gs = None

    def _soilwarn(self, varname: str) -> None:
        """Emit warning about soil moisture stress factor.

        The empirical soil moisture stress factor (Stocker et al. 2020) _can_ be
        used to back calculate realistic Jmax and Vcmax values. The
        pyrealm.PModel implementation does not do so and this helper function is
        used to warn users within property getter functions
        """

        if self.do_soilmstress:
            warn(
                f"pyrealm.PModel does not correct {varname} for empirical soil "
                "moisture effects on LUE."
            )

    def _check_estimated(self, varname: str) -> None:
        """Raise error when accessing unpopulated parameters.

        A helper function to raise an error when a user accesses a P Model
        parameter that has not yet been estimated via `estimate_productivity`.
        """
        if getattr(self, "_" + varname) is None:
            raise RuntimeError(f"{varname} not calculated: use estimate_productivity")

    @property
    def gpp(self) -> Union[float, np.ndarray]:
        """Fetch GPP if estimated."""
        self._check_estimated("gpp")

        return self._gpp

    @property
    def vcmax(self) -> Union[float, np.ndarray]:
        """Fetch V_cmax if estimated."""
        self._check_estimated("vcmax")
        self._soilwarn("vcmax")
        return self._vcmax

    @property
    def vcmax25(self) -> Union[float, np.ndarray]:
        """Fetch V_cmax25 if estimated."""
        self._check_estimated("vcmax25")
        self._soilwarn("vcmax25")
        return self._vcmax25

    @property
    def rd(self) -> Union[float, np.ndarray]:
        """Fetch dark respiration if estimated."""
        self._check_estimated("rd")
        self._soilwarn("rd")
        return self._rd

    @property
    def jmax(self) -> Union[float, np.ndarray]:
        """Fetch Jmax if estimated."""
        self._check_estimated("jmax")
        self._soilwarn("jmax")
        return self._jmax

    @property
    def gs(self) -> Union[float, np.ndarray]:
        """Fetch gs if estimated."""
        self._check_estimated("gs")
        self._soilwarn("gs")
        return self._gs

    def estimate_productivity(
        self, fapar: Union[float, np.ndarray] = 1, ppfd: Union[float, np.ndarray] = 1
    ):
        r"""Estimate productivity of P Model from absorbed irradiance.

        This method takes the light use efficiency and Vcmax per unit
        absorbed irradiance and populates the PModel instance with estimates
        of the following:

            * gpp: Gross primary productivity
            * rd: Dark respiration
            * vcmax: Maximum rate of carboxylation
            * vcmax25: Maximum rate of carboxylation at standard temperature
            * jmax: Maximum rate of electron transport.
            * gs: Stomatal conductance

        The functions finds the total absorbed irradiance (:math:`I_{abs}`) as
        the product of the photosynthetic photon flux density (PPFD, `ppfd`) and
        the fraction of absorbed photosynthetically active radiation (`fapar`).

        The default values of ``ppfd`` and ``fapar`` provide estimates of the
        variables above per unit absorbed irradiance.

        PPFD _must_ be provided in units of micromols per metre square per
        second (µmol m-2 s-1). This is required to ensure that values of
        :math:`J_{max}` and :math:`V_{cmax}` are also in µmol m-2 s-1.

        Args:
            fapar: the fraction of absorbed photosynthetically active radiation (-)
            ppfd: photosynthetic photon flux density (µmol m-2 s-1)
        """

        # Check input shapes against each other and an existing calculated value
        _ = check_input_shapes(ppfd, fapar, self.lue)

        # Calculate Iabs
        iabs = fapar * ppfd

        # GPP
        self._gpp = self.lue * iabs

        # V_cmax
        self._vcmax = self.kphio * iabs * self.optchi.mjoc * self.jmaxlim.f_v

        # V_cmax25 (vcmax normalized to pmodel_params.k_To)
        ftemp25_inst_vcmax = calc_ftemp_inst_vcmax(
            self.env.tc, pmodel_params=self.pmodel_params
        )
        self._vcmax25 = self._vcmax / ftemp25_inst_vcmax

        # Dark respiration at growth temperature
        ftemp_inst_rd = calc_ftemp_inst_rd(
            self.env.tc, pmodel_params=self.pmodel_params
        )
        self._rd = (
            self.pmodel_params.atkin_rd_to_vcmax
            * (ftemp_inst_rd / ftemp25_inst_vcmax)
            * self._vcmax
        )

        # Calculate Jmax
        self._jmax = 4 * self.kphio * iabs * self.jmaxlim.f_j

        # AJ and AC
        a_j = self.kphio * iabs * self.optchi.mj * self.jmaxlim.f_v
        a_c = self._vcmax * self.optchi.mc

        assim = np.minimum(a_j, a_c)

        if not self.do_soilmstress and not np.allclose(
            assim, self._gpp / self.pmodel_params.k_c_molmass, equal_nan=True
        ):
            warn("Assimilation and GPP are not identical")

        # Stomatal conductance
        self._gs = assim / (self.env.ca - self.optchi.ci)

    def __repr__(self):
        if self.do_soilmstress:
            stress = "Soil moisture"
        elif self.do_rootzonestress:
            stress = "Root zone"
        else:
            stress = "None"
        return (
            f"PModel("
            f"shape={self.shape}, "
            f"initial kphio={self.init_kphio}, "
            f"ftemp_kphio={self.do_ftemp_kphio}, "
            f"method_optchi={self.method_optchi}, "
            f"c4={self.c4}, "
            f"method_jmaxlim={self.method_jmaxlim}, "
            f"Water stress={stress})"
        )

    def summarize(self, dp: int = 2) -> None:
        """Print PModel summary.

        Prints a summary of the calculated values in a PModel instance
        including the mean, range and number of nan values. This will always
        show efficiency variables (LUE and IWUE) and productivity estimates are
        shown if :meth:`~pyrealm.pmodel.PModel.calculate_productivity` has been
        run.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [("lue", "g C mol-1"), ("iwue", "µmol mol-1")]

        if self._gpp is not None:
            attrs.extend(
                [
                    ("gpp", "µg C m-2 s-1"),
                    ("vcmax", "µmol m-2 s-1"),
                    ("vcmax25", "µmol m-2 s-1"),
                    ("rd", "µmol m-2 s-1"),
                    ("gs", "µmol m-2 s-1"),
                    ("jmax", "µmol m-2 s-1"),
                ]
            )

        summarize_attrs(self, attrs, dp=dp)


class CalcOptimalChi:
    r"""Estimate optimal leaf internal CO2 concentration.

    This class provides alternative approaches to calculating the optimal
    :math:`\chi` and :math:`\ce{CO2}` limitation factors. These values are:

    - The optimal ratio of leaf internal to ambient :math:`\ce{CO2}` partial
      pressure (:math:`\chi = c_i/c_a`).
    - The :math:`\ce{CO2}` limitation term for light-limited
      assimilation (:math:`m_j`).
    - The :math:`\ce{CO2}` limitation term for Rubisco-limited
      assimilation  (:math:`m_c`).

    The chosen method is automatically used to estimate these values when an
    instance is created - see the method documentation for individual details.

    The ratio of carboxylation to transpiration cost factors (``beta``, :math:`\beta`)
    is a key parameter in these methods. It is often held constant across cells but some
    methods (``lavergne20_c3`` and ``lavergne20_c4``) calculate :math:`beta` from
    environmental conditions. For this reason, the ``beta`` attribute records the values
    used in calculations.

    Args:
        env: An instance of PModelEnvironment providing the photosynthetic
            environment for the model.
        method: The method to use for estimating optimal :math:`\chi`, one
            of ``prentice14`` (default), ``lavergne20_c3``, ``c4``,
            ``c4_no_gamma`` or ``lavergne20_c4``.
        rootzonestress: This is an experimental feature to supply a root zone
            stress factor used as a direct penalty to :math:`\beta`.
        pmodel_params: An instance of
            :class:`~pyrealm.param_classes.PModelParams`.

    Attributes:
        env (PModelEnvironment): An instance of PModelEnvironment providing
            the photosynthetic environment for the model.
        method (str): one of ``prentice14``, ``lavergne20``, ``c4``,
            ``c4_no_gamma`` or ``lavergne20_c4``.
        beta (float): the ratio of carboxylation to transpiration cost factors.
        xi (float): defines the sensitivity of :math:`\chi` to the vapour
            pressure deficit and  is related to the carbon cost of water
            (Medlyn et al. 2011; Prentice et 2014)
        chi (float): the ratio of leaf internal to ambient :math:`\ce{CO2}`
            partial pressure (:math:`\chi`).
        mj (float): :math:`\ce{CO2}` limitation factor for light-limited
            assimilation (:math:`m_j`).
        mc (float): :math:`\ce{CO2}` limitation factor for RuBisCO-limited
            assimilation (:math:`m_c`).
        mjoc (float):  :math:`m_j/m_c` ratio

    Returns:
        An instance of :class:`CalcOptimalChi` where the :attr:`chi`,
        :attr:`mj`, :attr:`mc` and :attr:`mjoc` have been populated
        using the chosen method.

    """

    def __init__(
        self,
        env: PModelEnvironment,
        rootzonestress: Optional[Union[float, np.ndarray]] = None,
        method: str = "prentice14",
        pmodel_params: PModelParams = PModelParams(),
    ):
        # Store the PModelEnvironment
        self.env = env

        # Check rootzonestress conforms to the environment data
        if rootzonestress is not None:
            self.shape = check_input_shapes(env.ca, rootzonestress)
            self.rootzonestress = rootzonestress
            warn("The rootzonestress option is an experimental feature.")
        else:
            self.shape = env.shape
            self.rootzonestress = None

        # set attribute defaults
        self.beta = None
        self.xi = None
        self.chi = None
        self.ci = None
        self.mc = None
        self.mj = None
        self.mjoc = None

        # TODO: considerable overlap between methods here - could maybe bring
        #       more into init but probably clearer and easier to debug to keep
        #       complete method code within methods.

        # Identify and run the selected method
        self.pmodel_params = pmodel_params
        self.method = method

        # Check the method - return value shows C3/C4 but not needed here - and
        # then run that method to populate the instance
        _ = self._method_lookup(method)
        method_func = getattr(self, method)
        method_func()

    @staticmethod
    def _method_lookup(method: str) -> bool:
        """Check a CalcOptimalChi method.

        This function validates a provided string, identifying a method of
        CalcOptimalChi, and also reports if the method is C4 or not.

        Args:
            method: A provided method name for
                :class:`~pyrealm.pmodel.CalcOptimalChi`.

        Returns:
            A boolean indicating showing if the method uses the C3 pathway.
        """

        map = {
            "prentice14": False,
            "lavergne20_c3": False,
            "c4": True,
            "c4_no_gamma": True,
            "lavergne20_c4": True,
        }

        is_c4 = map.get(method)

        if is_c4 is None:
            raise ValueError(f"CalcOptimalChi: method argument '{method}' invalid.")

        return is_c4

    def __repr__(self):

        return f"CalcOptimalChi(shape={self.shape}, method={self.method})"

    def prentice14(self) -> None:
        r"""Calculate :math:`\chi` for C3 plants following :cite:`Prentice:2014bc`.

        Optimal :math:`\chi` is calculated following Equation 8 in
        :cite:`Prentice:2014bc`:

          .. math:: :nowrap:

            \[
                \begin{align*}
                    \chi &= \Gamma^{*} / c_a + (1- \Gamma^{*} / c_a)
                        \xi / (\xi + \sqrt D ), \text{where}\\
                    \xi &= \sqrt{(\beta (K+ \Gamma^{*}) / (1.6 \eta^{*}))}
                \end{align*}
            \]

        The :math:`\ce{CO2}` limitation term of light use efficiency
        (:math:`m_j`) is calculated following Equation 3 in :cite:`Wang:2017go`:

        .. math::

            m_j = \frac{c_a - \Gamma^{*}}
                       {c_a + 2 \Gamma^{*}}

        Finally,  :math:`m_c` is calculated, following Equation 7 in
        :cite:`Stocker:2020dh`, as:

        .. math::

            m_c = \frac{c_i - \Gamma^{*}}{c_i + K},

        where :math:`K` is the Michaelis Menten coefficient of Rubisco-limited
        assimilation.

        Examples:
            >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
            >>> vals = CalcOptimalChi(env=env)
            >>> round(vals.chi, 5)
            0.69435
            >>> round(vals.mc, 5)
            0.33408
            >>> round(vals.mj, 5)
            0.7123
            >>> round(vals.mjoc, 5)
            2.13211
        """

        # TODO: Docstring - equation for m_j previously included term now
        # removed from code:
        #     + 3 \Gamma^{*}
        #               \sqrt{\frac{1.6 D \eta^{*}}{\beta(K + \Gamma^{*})}}

        # Replace missing rootzonestress with 1
        self.rootzonestress = self.rootzonestress or 1.0

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.pmodel_params.beta_cost_ratio_prentice14
        self.xi = np.sqrt(
            (self.beta * self.rootzonestress * (self.env.kmm + self.env.gammastar))
            / (1.6 * self.env.ns_star)
        )

        self.chi = self.env.gammastar / self.env.ca + (
            1.0 - self.env.gammastar / self.env.ca
        ) * self.xi / (self.xi + np.sqrt(self.env.vpd))

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mj = (self.ci - self.env.gammastar) / (self.ci + 2 * self.env.gammastar)
        self.mc = (self.ci - self.env.gammastar) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def lavergne20_c3(self) -> None:
        r"""Calculate soil moisture corrected :math:`\chi` for C3 plants.

        This method calculates the unit cost ratio $\beta$ as a function of soil
        moisture ($\theta$, m3 m-3), following :cite:`lavergne:2020a`:

          .. math:: :nowrap:

            \[
                \beta = e ^ {b\theta + a},
            \]

        The coefficients are experimentally derived values with defaults taken
        from Figure 6a of :cite:`lavergne:2020a` (:math:`a`,
        :meth:`~pyrealm.param_classes.PModelParams.lavergne_2020_a`; :math:`b`,
        :meth:`~pyrealm.param_classes.PModelParams.lavergne_2020_b`).

        Values of :math:`\chi` and other predictions are then calculated as in
        :meth:`~pyrealm.pmodel.CalcOptimalChi.prentice14`. This method requires
        that `env` includes estimates of :math:`\theta` and  is incompatible with
        the `rootzonestress` approach.

        Examples:
            >>> env = PModelEnvironment(tc=20, patm=101325, co2=400,
            ...                         vpd=1000, theta=0.5)
            >>> vals = CalcOptimalChi(env=env, method='lavergne20_c3')
            >>> round(vals.beta, 5)
            224.75255
            >>> round(vals.chi, 5)
            0.73663
            >>> round(vals.mc, 5)
            0.34911
            >>> round(vals.mj, 5)
            0.7258
            >>> round(vals.mjoc, 5)
            2.07901
        """

        # This method needs theta
        if self.env.theta is None:
            raise RuntimeError(
                "Method `lavergne20_c3` requires soil moisture in the PModelEnvironment"
            )

        # Calculate beta as a function of theta
        self.beta = np.exp(
            self.pmodel_params.lavergne_2020_b_c3 * self.env.theta
            + self.pmodel_params.lavergne_2020_a_c3
        )

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.xi = np.sqrt(
            (self.beta * (self.env.kmm + self.env.gammastar)) / (1.6 * self.env.ns_star)
        )

        self.chi = self.env.gammastar / self.env.ca + (
            1.0 - self.env.gammastar / self.env.ca
        ) * self.xi / (self.xi + np.sqrt(self.env.vpd))

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mj = (self.ci - self.env.gammastar) / (self.ci + 2 * self.env.gammastar)
        self.mc = (self.ci - self.env.gammastar) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def lavergne20_c4(self) -> None:
        r"""Calculate soil moisture corrected :math:`\chi` for C4 plants.

        This method calculates :math:`\beta` as a function of soil moisture following
        the equation described in the :meth:`~pyrealm.pmodel.CalcOptimalChi.lavergne20`
        method.  However, the default coefficients of the moisture scaling from
        :cite:`lavergne:2020a` for C3 plants are adjusted to match the theoretical
        expectation that :math:`\beta` for C4 plants is nine times smaller than
        :math:`\beta` for C3 plants (see :meth:`~pyrealm.pmodel.CalcOptimalChi.c4`):
        :math:`b` is unchanged but :math:`a_{C4} = a_{C3} - log(9)`.

        Following the calculation of :math:`\beta`, this method then follows the
        calculations described in :meth:`~pyrealm.pmodel.CalcOptimalChi.c4_no_gamma`:
        :math:`m_j = 1.0`  because photorespiration is negligible, but :math:`m_c` and
        hence :math:`m_{joc}` are calculated.

        Note:

        This is an **experimental approach**. The research underlying
        :cite:`lavergne:2020a`, found **no relationship** between C4 :math:`\beta`
        values and soil moisture in leaf gas exchange measurements.

        Examples:
            >>> env = PModelEnvironment(tc=20, patm=101325, co2=400,
            ...                         vpd=1000, theta=0.5)
            >>> vals = CalcOptimalChi(env=env, method='lavergne20_c4')
            >>> round(vals.beta, 5)
            24.97251
            >>> round(vals.chi, 5)
            0.44432
            >>> round(vals.mc, 5)
            0.28091
            >>> round(vals.mj, 5)
            1.0
            >>> round(vals.mjoc, 5)
            3.55989
        """

        # Warn that this is experimental
        warn(
            "The lavergne20_c4 method is experimental, see the method documentation",
            ExperimentalFeatureWarning,
        )

        # This method needs theta
        if self.env.theta is None:
            raise RuntimeError(
                "Method `lavergne20_c4` requires soil moisture in the PModelEnvironment"
            )

        # Calculate beta as a function of theta
        self.beta = np.exp(
            self.pmodel_params.lavergne_2020_b_c4 * self.env.theta
            + self.pmodel_params.lavergne_2020_a_c4
        )

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        self.xi = np.sqrt((self.beta * self.env.kmm) / (1.6 * self.env.ns_star))
        self.chi = self.xi / (self.xi + np.sqrt(self.env.vpd))

        # mj is equal to 1 as gammastar is null
        if self.shape == 1:
            self.mj = 1.0
        else:
            self.mj = np.ones(self.shape)

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mc = (self.ci) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def c4(self) -> None:
        r"""Estimate :math:`\chi` for C4 plants following :cite:`Prentice:2014bc`.

        Optimal :math:`\chi` is calculated as in
        :meth:`~pyrealm.pmodel.CalcOptimalChi.prentice14`, but using a C4
        specific estimate of the unit cost ratio :math:`\beta`, specified in
        :meth:`~pyrealm.param_classes.PModelParams.beta_cost_ratio_c4`.
        The default value :math:`\beta = 146 /  9 \approx 16.222`. This is
        derived from estimates of the :math:`g_1` parameter for C3 and C4 plants in
        :cite:`Lin:2015wh`  and :cite:`DeKauwe:2015im`, which have a C3/C4
        ratio of around 3. Given that :math:`g_1 \equiv \xi \propto \surd\beta`,
        a reasonable default for C4 plants is that :math:`\beta_{C4} \approx
        \beta_{C3} / 9`.

        This method  sets :math:`m_j = m_c = m_{joc} = 1.0` to capture the
        boosted :math:`\ce{CO2}` concentrations at the chloropolast in C4
        photosynthesis.

        Examples:
            >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
            >>> vals = CalcOptimalChi(env=env, method='c4')
            >>> round(vals.chi, 5)
            0.44967
            >>> round(vals.mj, 1)
            1.0
            >>> round(vals.mc, 1)
            1.0
        """

        # Replace missing rootzonestress with 1
        self.rootzonestress = self.rootzonestress or 1.0

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.beta = self.pmodel_params.beta_cost_ratio_c4
        self.xi = np.sqrt(
            (self.beta * self.rootzonestress * (self.env.kmm + self.env.gammastar))
            / (1.6 * self.env.ns_star)
        )

        self.chi = self.env.gammastar / self.env.ca + (
            1.0 - self.env.gammastar / self.env.ca
        ) * self.xi / (self.xi + np.sqrt(self.env.vpd))

        self.ci = self.chi * self.env.ca

        # These values need to retain any
        # dimensions of the original inputs - if ftemp_kphio is set to 1.0
        # (i.e. no temperature correction) then the dimensions of tc are lost.
        if self.shape == 1:
            self.mc = 1.0
            self.mj = 1.0
            self.mjoc = 1.0
        else:
            self.mc = np.ones(self.shape)
            self.mj = np.ones(self.shape)
            self.mjoc = np.ones(self.shape)

    def c4_no_gamma(self) -> None:
        r"""Calculate optimal chi assuming negligible photorespiration.

        This method assumes that photorespiration (:math:`\Gamma^\ast`) is
        negible for C4 plants. This simplifies the calculation of :math:`\xi`
        and :math:`\chi` compared to :meth:`~pyrealm.pmodel.CalcOptimalChi.c4`,
        but uses the same C4 specific estimate of the unit cost ratio
        :math:`\beta`,
        :meth:`~pyrealm.param_classes.PModelParams.beta_cost_ratio_c4`.

          .. math:: :nowrap:

            \[
                \begin{align*}
                    \chi &= \xi / (\xi + \sqrt D ), \text{where}\\ \xi &=
                    \sqrt{(\beta  K) / (1.6 \eta^{*}))}
                \end{align*}
            \]

        In addition, :math:`m_j = 1.0`  because photorespiration is negligible
        in C4 photosynthesis, but :math:`m_c` and hence :math:`m_{joc}` are
        calculated, not set to one.

          .. math:: :nowrap:

            \[
                \begin{align*}
                    m_c &= \dfrac{\chi}{\chi + \frac{K}{c_a}}
                \end{align*}
            \]

        Examples:
            >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
            >>> vals = CalcOptimalChi(env=env, method='c4_no_gamma')
            >>> round(vals.chi, 5)
            0.3919
            >>> round(vals.mj, 1)
            1.0
            >>> round(vals.mc, 1)
            0.3
        """

        # Replace missing rootzonestress with 1
        self.rootzonestress = self.rootzonestress or 1.0

        # Calculate chi and xi as in Prentice 14 but removing gamma terms.
        self.beta = self.pmodel_params.beta_cost_ratio_c4
        self.xi = np.sqrt(
            (self.beta * self.rootzonestress * self.env.kmm) / (1.6 * self.env.ns_star)
        )

        self.chi = self.xi / (self.xi + np.sqrt(self.env.vpd))

        # version 3 as in Scott & Smith (2022)
        # beta_c4 = 166
        # self.xi = np.sqrt((beta_c4 *self.env.kp_c4) / (1.6 * self.env.ns_star))
        # self.chi = self.xi /(self.xi + np.sqrt(self.env.vpd))

        # mj is equal to 1 as gammastar is null
        if self.shape == 1:
            self.mj = 1.0
        else:
            self.mj = np.ones(self.shape)

        # Calculate m and mc and m/mc
        self.ci = self.chi * self.env.ca
        self.mc = (self.ci) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc

    def summarize(self, dp: int = 2) -> None:
        """Print CalcOptimalChi summary.

        Prints a summary of the variables calculated within an instance
        of CalcOptimalChi including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [
            ("xi", " Pa ^ (1/2)"),
            ("chi", "-"),
            ("mc", "-"),
            ("mj", "-"),
            ("mjoc", "-"),
        ]
        summarize_attrs(self, attrs, dp=dp)


class JmaxLimitation:
    r"""Estimate JMax limitation.

    This class calculates two factors (:math:`f_v` and :math:`f_j`) used to
    implement :math:`V_{cmax}` and :math:`J_{max}` limitation of photosynthesis.
    Three methods are currently implemented:

        * ``simple``: applies the 'simple' equations with no limitation. The
          alias ``none`` is also accepted.
        * ``wang17``: applies the framework of :cite:`Wang:2017go`.
        * ``smith19``: applies the framework of :cite:`Smith:2019dv`

    Note that :cite:`Smith:2019dv` defines :math:`\phi_0` as the quantum
    efficiency of electron transfer, whereas :mod:`pyrealm.PModel` defines
    :math:`\phi_0` as the quantum efficiency of photosynthesis, which is 4 times
    smaller. This is why the factors here are a factor of 4 greater than Eqn 15
    and 17 in :cite:`Smith:2019dv`.

    Arguments:
        optchi: an instance of :class:`CalcOptimalChi` providing the :math:`\ce{CO2}`
            limitation term of light use efficiency (:math:`m_j`) and the
            :math:`\ce{CO2}` limitation term for Rubisco assimilation (:math:`m_c`).
        method: method to apply :math:`J_{max}` limitation (default: ``wang17``,
            or ``smith19`` or ``none``)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Attributes:
        f_j (float): :math:`J_{max}` limitation factor, calculated using the method.
        f_v (float): :math:`V_{cmax}` limitation factor, calculated using the method.
        omega (float): component of :math:`J_{max}` calculation (:cite:`Smith:2019dv`).
        omega_star (float):  component of :math:`J_{max}` calculation
            (:cite:`Smith:2019dv`).

    Examples:
        >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000)
        >>> optchi = CalcOptimalChi(env)
        >>> simple = JmaxLimitation(optchi, method='simple')
        >>> simple.f_j
        1.0
        >>> simple.f_v
        1.0
        >>> wang17 = JmaxLimitation(optchi, method='wang17')
        >>> round(wang17.f_j, 5)
        0.66722
        >>> round(wang17.f_v, 5)
        0.55502
        >>> smith19 = JmaxLimitation(optchi, method='smith19')
        >>> round(smith19.f_j, 5)
        1.10204
        >>> round(smith19.f_v, 5)
        0.75442
    """

    # TODO - apparent incorrectness of wang and smith methods with _ca_ variation,
    #        work well with varying temperature but not _ca_ variation (or
    #        e.g. elevation gradient David Sandoval, REALM meeting, Dec 2020)

    def __init__(
        self,
        optchi: CalcOptimalChi,
        method: str = "wang17",
        pmodel_params: PModelParams = PModelParams(),
    ):

        self.shape = check_input_shapes(optchi.mj)

        self.optchi = optchi
        self.method = method
        self.pmodel_params = pmodel_params
        self.f_j = None
        self.f_m = None
        self.omega = None
        self.omega_star = None

        all_methods = {
            "wang17": self.wang17,
            "smith19": self.smith19,
            "simple": self.simple,
            "none": self.simple,
        }

        # Catch method errors.
        if self.method == "c4":
            raise ValueError(
                "This class does not implement a fixed method for C4 photosynthesis."
                "To replicate rpmodel choose method_optchi='c4' and method='simple'"
            )
        elif self.method not in all_methods:
            raise ValueError(f"JmaxLimitation: method argument '{method}' invalid.")

        # Use the selected method to calculate limitation factors
        this_method = all_methods[self.method]
        this_method()

    def __repr__(self):

        return f"JmaxLimitation(shape={self.shape})"

    def wang17(self):
        r"""Calculate limitation factors following :cite:`Wang:2017go`.

        These factors are described in Equation 49 of :cite:`Wang:2017go` as the
        square root term at the end of that equation:

            .. math::
                :nowrap:

                \[
                    \begin{align*}
                    f_v &=  \sqrt{ 1 - \frac{c^*}{m} ^{2/3}} \\
                    f_j &=  \sqrt{\frac{m}{c^*} ^{2/3} -1 } \\
                    \end{align*}
                \]

        The variable :math:`c^*` is a cost parameter for maintaining :math:`J_{max}`
        and is set in `pmodel_params.wang_c`.
        """

        # Calculate √ {1 – (c*/m)^(2/3)} (see Eqn 2 of Wang et al 2017) and
        # √ {(m/c*)^(2/3) - 1} safely, both are undefined where m <= c*.
        vals_defined = np.greater(self.optchi.mj, self.pmodel_params.wang17_c)

        self.f_v = np.sqrt(
            1 - (self.pmodel_params.wang17_c / self.optchi.mj) ** (2.0 / 3.0),
            where=vals_defined,
        )
        self.f_j = np.sqrt(
            (self.optchi.mj / self.pmodel_params.wang17_c) ** (2.0 / 3.0) - 1,
            where=vals_defined,
        )

        # Backfill undefined values
        if isinstance(self.f_v, np.ndarray):
            self.f_j[np.logical_not(vals_defined)] = np.nan
            self.f_v[np.logical_not(vals_defined)] = np.nan
        elif not vals_defined:
            self.f_j = np.nan
            self.f_v = np.nan

    def smith19(self):
        r"""Calculate limitation factors following :cite:`Smith:2019dv`.

        The values are calculated as:

        .. math::
            :nowrap:

            \[
                \begin{align*}
                f_v &=  \frac{\omega^*}{2\theta} \\
                f_j &=  \omega\\
                \end{align*}
            \]

        where,

        .. math::
            :nowrap:

            \[
                \begin{align*}
                \omega &= (1 - 2\theta) + \sqrt{(1-\theta)
                    \left(\frac{1}{\frac{4c}{m}(1 -
                    \theta\frac{4c}{m})}-4\theta\right)}\\
                \omega^* &= 1 + \omega - \sqrt{(1 + \omega) ^2 -4\theta\omega}
                \end{align*}
            \]

        given,
            * :math:`\theta`, (`pmodel_params.smith19_theta`) captures the
              curved relationship between light intensity and photosynthetic
              capacity, and
            * :math:`c`, (`pmodel_params.smith19_c_cost`) as a cost parameter
              for maintaining :math:`J_{max}`, equivalent to :math:`c^\ast = 4c`
              in the :meth:`~pyrealm.pmodel.JmaxLimitation.wang17` method.
        """

        # Adopted from Nick Smith's code:
        # Calculate omega, see Smith et al., 2019 Ecology Letters  # Eq. S4
        theta = self.pmodel_params.smith19_theta
        c_cost = self.pmodel_params.smith19_c_cost

        # simplification terms for omega calculation
        cm = 4 * c_cost / self.optchi.mj
        v = 1 / (cm * (1 - self.pmodel_params.smith19_theta * cm)) - 4 * theta

        # account for non-linearities at low m values. This code finds
        # the roots of a quadratic function that is defined purely from
        # the scalar theta, so will always be a scalar. The first root
        # is then used to set a filter for calculating omega.

        cap_p = (((1 / 1.4) - 0.7) ** 2 / (1 - theta)) + 3.4
        aquad = -1
        bquad = cap_p
        cquad = -(cap_p * theta)
        roots = np.polynomial.polynomial.polyroots([aquad, bquad, cquad])

        # factors derived as in Smith et al., 2019
        m_star = (4 * c_cost) / roots[0].real
        omega = np.where(
            self.optchi.mj < m_star,
            -(1 - (2 * theta)) - np.sqrt((1 - theta) * v),
            -(1 - (2 * theta)) + np.sqrt((1 - theta) * v),
        )

        # np.where _always_ returns an array, so catch scalars
        self.omega = omega.item() if np.ndim(omega) == 0 else omega

        self.omega_star = (
            1.0
            + self.omega
            - np.sqrt((1.0 + self.omega) ** 2 - (4.0 * theta * self.omega))  # Eq. 18
        )

        # Effect of Jmax limitation - note scaling here. Smith et al use
        # phi0 as as the quantum efficiency of electron transport, which is
        # 4 times our definition of phio0 as the quantum efficiency of photosynthesis.
        # So omega*/8 theta and omega / 4 are scaled down here  by a factor of 4.
        self.f_v = self.omega_star / (2.0 * theta)
        self.f_j = self.omega

    def simple(self):
        """Apply the 'simple' form of the equations.

        This method allows the 'simple' form of the equations to be calculated
        by setting :math:`f_v = f_j = 1`.
        """

        # Set Jmax limitation to unity - could define as 1.0 in __init__ and
        # pass here, but setting explicitly within the method for clarity.
        self.f_v = 1.0
        self.f_j = 1.0


class CalcCarbonIsotopes:
    r"""Calculate :math:`\ce{CO2}` isotopic discrimination.

    This class estimates the fractionation of atmospheric CO2 by photosynthetic
    pathways to calculate the isotopic compositions and discrimination given the
    predicted optimal chi from a :class:`~pyrealm.pmodel.PModel` instance.

    Discrimination against carbon 13 (:math:`\Delta\ce{^{13}C}`)  is calculated
    using C3 and C4 pathways specific methods, and then discrimination against
    carbon 14 is estimated as :math:`\Delta\ce{^{14}C} \approx 2 \times
    \Delta\ce{^{13}C}` (:cite:`graven:2020a`). For C3 plants,
    :math:`\Delta\ce{^{13}C}` is calculated both including and excluding
    photorespiration, but these are assumed to be equal for C4 plants. The class
    also reports the isotopic composition of leaves and wood.

    Args:
        pmodel: A :class:`~pyrealm.pmodel.PModel` instance providing the
            photosynthetic pathway and estimated optimal chi.
        d13CO2: Atmospheric isotopic ratio for Carbon 13
            (:math:`\delta\ce{^{13}C}`, permil).
        D14CO2: Atmospheric isotopic ratio for Carbon 14
            (:math:`\Delta\ce{^{14}C}`, permil).
        params: An instance of :class:`~pyrealm.param_classes.IsotopesParams`,
            parameterizing the calculations.

    Attributes:
        Delta13C_simple: discrimination against carbon 13
            (:math:`\Delta\ce{^{13}C}`, permil) excluding photorespiration.
        Delta13C: discrimination against carbon 13
            (:math:`\Delta\ce{^{13}C}`, permil) including photorespiration.
        Delta14C: discrimination against carbon 14
            (:math:`\Delta\ce{^{14}C}`, permil) including photorespiration.
        d13C_leaf: isotopic ratio of carbon 13 in leaves
            (:math:`\delta\ce{^{13}C}`, permil).
        d14C_leaf: isotopic ratio of carbon 14 in leaves
            (:math:`\delta\ce{^{14}C}`, permil).
        d13C_wood: isotopic ratio of carbon 13 in wood
            (:math:`\delta\ce{^{13}C}`, permil), given a parameterized
            post-photosynthetic fractionation.
    """

    def __init__(
        self,
        pmodel: PModel,
        d13CO2: Union[float, np.ndarray],
        D14CO2: Union[float, np.ndarray],
        params: IsotopesParams = IsotopesParams(),
    ):

        # Check inputs are congruent
        _ = check_input_shapes(pmodel.env.tc, d13CO2, D14CO2)

        self.params = params
        self.shape = pmodel.shape
        self.c4 = pmodel.c4

        self.Delta13C_simple = None
        self.Delta13C = None
        self.Delta14C = None
        self.d13C_leaf = None
        self.d14C_leaf = None
        self.d13C_wood = None

        # Could store pmodel, d13CO2, D14CO2 in instance, but really not needed
        # so try and keep this class simple with a minimum of attributes.
        # TODO: map methods for delta13C to C3 and C4.

        if self.c4:
            self.calc_c4_discrimination(pmodel)
        else:
            self.calc_c3_discrimination(pmodel)

        # 14C discrimination is twice the 13C discrimination (Graven et al. 2020)
        self.Delta14C = self.Delta13C * 2

        # Isotopic composition of leaf
        self.d13C_leaf = (d13CO2 - self.Delta13C) / (1 + self.Delta13C / 1000)
        self.d14C_leaf = (D14CO2 - self.Delta14C) / (1 + self.Delta14C / 1000)

        # Isotopic composition of wood considering post-photosynthetic fractionation:
        self.d13C_wood = self.d13C_leaf + self.params.frank_postfrac

    def __repr__(self):

        return f"CalcCarbonIsotopes(shape={self.shape}, method={self.c4})"

    def calc_c4_discrimination(self, pmodel):
        r"""Calculate C4 isotopic discrimination.

        In this method, :math:`\delta\ce{^{13}C}` is calculated from optimal
        :math:`\chi` using an empirical relationship estimated by
        :cite:`lavergne:2022a`.

        Examples:
            >>> ppar = PModelParams(beta_cost_ratio_c4=35)
            >>> env = PModelEnvironment(tc=20, patm=101325, co2=400,
            ...                         vpd=1000, pmodel_params=ppar)
            >>> mod_c4 = PModel(env, method_optchi='c4_no_gamma')
            >>> mod_c4_delta = CalcCarbonIsotopes(mod_c4, d13CO2= -8.4, D14CO2 = 19.2)
            >>> round(mod_c4_delta.Delta13C, 4)
            5.6636
            >>> round(mod_c4_delta.d13C_leaf, 4)
            -13.9844
        """

        # Equation from C3/C4 paper
        self.Delta13C_simple = (
            self.params.lavergne_delta13_a
            + self.params.lavergne_delta13_b * pmodel.optchi.chi
        )
        self.Delta13C = self.Delta13C_simple

    def calc_c4_discrimination_vonC(self, pmodel):
        r"""Calculate C4 isotopic discrimination.

        In this method, :math:`\delta\ce{^{13}C}` is calculated from optimal
        :math:`\chi` following Equation 1 in :cite:`voncaemmerer:2014a`.

        This method is not yet reachable - it needs a method selection argument to
        switch approaches and check C4 methods are used with C4 pmodels. The
        method is preserving experimental code provided by Alienor Lavergne. A
        temperature sensitive correction term is provided in commented code but
        not used.

        Examples:
            >>> ppar = PModelParams(beta_cost_ratio_c4=35)
            >>> env = PModelEnvironment(tc=20, patm=101325, co2=400,
            ...                         vpd=1000, pmodel_params=ppar)
            >>> mod_c4 = PModel(env, method_optchi='c4_no_gamma')
            >>> mod_c4_delta = CalcCarbonIsotopes(mod_c4, d13CO2= -8.4, D14CO2 = 19.2)
            >>> # round(mod_c4_delta.Delta13C, 4)  # NOT CHECKED 5.2753
            >>> # round(mod_c4_delta.d13C_leaf, 4)  # NOT CHECKED -13.6036
        """

        warn("This method is experimental code from Alienor Lavergne")

        # 13C discrimination (‰): von Caemmerer et al. (2014) Eq. 1
        self.Delta13C_simple = (
            self.params.farquhar_a
            + (
                self.params.vonCaemmerer_b4
                + (self.params.farquhar_b - self.params.vonCaemmerer_s)
                * self.params.vonCaemmerer_phi
                - self.params.farquhar_a
            )
            * pmodel.optchi.chi
        )

        # Equation A5 from von Caemmerer et al. (2014)
        # b4 = (-9.483 * 1000) / (273 + self.tc) + 23.89 + 2.2
        # b4 = self.pmodel_params.vonCaemmerer_b4

        self.Delta13C = self.Delta13C_simple

    def calc_c3_discrimination(self, pmodel):
        r"""Calculate C3 isotopic discrimination.

        This method calculates the isotopic discrimination for
        :math:`\Delta\ce{^{13}C}` both with and without the photorespiratory
        effect following :cite:`farquhar:1982a`.

        Examples:
            >>> env = PModelEnvironment(tc=20, patm=101325, co2=400,
            ...                         vpd=1000, theta=0.4)
            >>> mod_c3 = PModel(env, method_optchi='lavergne20_c3')
            >>> mod_c3_delta = CalcCarbonIsotopes(mod_c3, d13CO2= -8.4, D14CO2 = 19.2)
            >>> round(mod_c3_delta.Delta13C, 4)
            20.4056
            >>> round(mod_c3_delta.d13C_leaf, 4)
            -28.2296
        """

        # 13C discrimination (permil): Farquhar et al. (1982)
        # Simple
        self.Delta13C_simple = (
            self.params.farquhar_a
            + (self.params.farquhar_b - self.params.farquhar_a) * pmodel.optchi.chi
        )

        # with photorespiratory effect:
        self.Delta13C = (
            self.params.farquhar_a
            + (self.params.farquhar_b2 - self.params.farquhar_a) * pmodel.optchi.chi
            - self.params.farquhar_f * pmodel.env.gammastar / pmodel.env.ca
        )

    def summarize(self, dp=2) -> None:
        """Print CalcCarbonIsotopes summary.

        Prints a summary of the variables calculated within an instance
        of CalcCarbonIsotopes including the mean, range and number of nan
        values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [
            ("Delta13C_simple", "permil"),
            ("Delta13C", "permil"),
            ("Delta14C", "permil"),
            ("d13C_leaf", "permil"),
            ("d14C_leaf", "permil"),
            ("d13C_wood", "permil"),
        ]

        summarize_attrs(self, attrs, dp=dp)


class C3C4Competition:
    r"""Implementation of the C3/C4 competition model.

    This class provides an implementation of the calculations of C3/C4
    competition, described by :cite:`lavergne:2020a`. The key inputs ``ggp_c3``
    and ``gpp_c4`` are gross primary productivity (GPP) estimates for C3 or C4
    pathways `alone`  using the :class:`~pyrealm.pmodel.PModel`

    These estimates are used to calculate the relative advantage of C4 over C3
    photosynthesis (:math:`A_4`), the expected fraction of C4 plants in the
    community (:math:`F_4`) and hence fraction of GPP from C4 plants as follows:

    1. The proportion advantage in GPP for C4 plants is calculated as:

        .. math::
            :nowrap:

            \[
            A_4 = \frac{\text{GPP}_{C4} - \text{GPP}_{C3}}{\text{GPP}_{C3}}
            \]

    2. The proportion GPP advantage :math:`A_4` is converted to an expected
       fraction of C4 :math:`F_4` plants using a logistic equation of
       :math:`A_4`, where :math:`A_4` is first modulated by percentage tree
       cover (TC):

        .. math::
            :nowrap:

            \[
                \begin{align*}
                    A_4^\prime &= \frac{A_4}{e^ {1 / 1 + \text{TC}}} \\
                    F_4 &= \frac{1}{1 + e^{k A_4^\prime} - q}
                \end{align*}
            \]

        The parameters are set in the ``params`` instance and are the slope of the
        equation (:math:`k`, ``adv_to_frac_k``) and :math:`A_4` value at the
        midpoint of the curve (:math:`q`, ``adv_to_frac_q``).

    3. A model of tree cover from C3 trees is then used to correct for shading
       of C4 plants due to canopy closure, even when C4 photosynthesis is
       advantagious. The estimated tree cover function is:

        .. math::
            :nowrap:

                \[
                    TC(\text{GPP}_{C3}) = a \cdot \text{GPP}_{C3} ^ b - c
                \]

       with parameters set in the `params` instance (:math:`a`, ``gpp_to_tc_a``;
       :math:`b`, ``gpp_to_tc_b``; :math:`c`, ``gpp_to_tc_c``). The proportion of
       GPP from C3 trees (:math:`h`) is then estimated using the predicted tree
       cover in locations relative to a threshold GPP value (:math:`\text{GPP}_{CLO}`,
       `c3_forest_closure_gpp`) above which canopy closure occurs. The value of
       :math:`h` is clamped in :math:`[0, 1]`:

        .. math::
            :nowrap:

                \[
                    h = \max\left(0, \min\left(
                        \frac{TC(\text{GPP}_{C3})}{TC(\text{GPP}_{CLO})}\right),
                        1 \right)
                \]

       The C4 fraction is then discounted as :math:`F_4 = F_4 (1 - h)`.

    4. Two masks are applied. First, :math:`F_4 = 0` in locations where the
       mean  air temperature of the coldest month is too low for C4 plants.
       Second, :math:`F_4` is set as unknown for croplands, where the fraction
       is set by agricultural management, not competition.

    Args:
        gpp_c3: Total annual GPP (gC m-2 yr-1) from C3 plants alone.
        gpp_c4: Total annual GPP (gC m-2 yr-1) from C4 plants alone.
        treecover: Percentage tree cover (%).
        below_t_min: A boolean mask, temperatures too low for C4 plants.
        cropland: A boolean mask indicating cropland locations.
        params: An instance of :class:`~pyrealm.param_classes.C3C4Params`
            providing parameterisation for the competition model.

    Attributes:
        gpp_adv_c4: The proportional advantage in GPP of C4 over C3 plants
        frac_c4: The estimated fraction of C4 plants.
        gpp_c3_contrib: The estimated contribution of C3 plants to GPP (gC m-2 yr-1)
        gpp_c4_contrib: The estimated contribution of C4 plants to GPP (gC m-2 yr-1)
    """

    # Design Notes: see paper Lavergne et al. (submitted).
    #
    # DO (24/05/2022): I have separated out the functions for different steps
    # into private methods, partly to keep the code cleaner, partly with a
    # slightly hopeful idea that future users could substitute these functions
    # via subclassing, but _mostly_ because being able to access these functions
    # independently makes it much easier to document the steps.

    # TODO - could accept PModel instances for gpp_c3 and gpp_c4 and auto-scale
    #        gpp and check that they are c3 and c4 models.
    #      - Would also allow the estimate isotopic discrimination to work
    #        automatically.
    #      - Axis argument to aggregate values along a time axis?
    #        nansum for gpp  and nanmean for  DeltaC13/4_alone.

    def __init__(
        self,
        gpp_c3: Union[float, np.ndarray],
        gpp_c4: Union[float, np.ndarray],
        treecover: Union[float, np.ndarray],
        below_t_min: Union[float, np.ndarray],
        cropland: Union[float, np.ndarray],
        params: C3C4Params = C3C4Params(),
    ):

        # Check inputs are congruent
        self.shape = check_input_shapes(
            gpp_c3, gpp_c4, treecover, cropland, below_t_min
        )
        self.params = params

        # Step 1: calculate the percentage advantage in GPP of C4 plants from
        # annual total GPP estimates for C3 and C4 plants. This uses use
        # np.full to handle division by zero without raising warnings
        gpp_adv_c4 = np.full(self.shape, np.nan)
        self.gpp_adv_c4 = np.divide(
            gpp_c4 - gpp_c3, gpp_c3, out=gpp_adv_c4, where=gpp_c3 > 0
        )

        # Step 2: calculate the initial C4 fraction based on advantage modulated
        # by treecover.
        frac_c4 = self._convert_advantage_to_c4_fraction(treecover=treecover)

        # Step 3: calculate the proportion of trees shading C4 plants, scaling
        # the predicted GPP to kilograms.
        prop_trees = self._calculate_tree_proportion(gppc3=gpp_c3 / 1000)
        frac_c4 = frac_c4 * (1 - prop_trees)

        # Step 4: remove areas below minimum temperature
        frac_c4[below_t_min] = 0

        # Step 5: remove cropland areas
        frac_c4[cropland] = np.nan

        self.frac_c4 = frac_c4

        self.gpp_c3_contrib = gpp_c3 * (1 - self.frac_c4)
        self.gpp_c4_contrib = gpp_c4 * self.frac_c4

        # Define attributes used elsewhere
        self.Delta13C_C3 = None
        self.Delta13C_C4 = None
        self.d13C_C3 = None
        self.d13C_C4 = None

    def __repr__(self):

        return f"C3C4competition(shape={self.shape})"

    def _convert_advantage_to_c4_fraction(
        self, treecover: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert C4 GPP advantage to C4 fraction.

        This method calculates an initial estimate of the fraction of C4 plants
        based on the proportional GPP advantage from C4 photosynthesis. The
        conversion is modulated by the proportion treecover.

        Args:
            treecover: The proportion tree cover at modelled locations.

        Returns:
            The estimated C4 fraction given the estimated C4 GPP advantage and
            tree cover.
        """

        frac_c4 = 1.0 / (
            1.0
            + np.exp(
                -self.params.adv_to_frac_k
                * (
                    (self.gpp_adv_c4 / np.exp(1 / (1 + treecover)))
                    - self.params.adv_to_frac_q
                )
            )
        )

        return frac_c4

    def _calculate_tree_proportion(self, gppc3) -> Union[float, np.ndarray]:
        """Calculate the proportion of GPP from C3 trees.

        This method calculates the proportional impact of forest closure by C3
        trees on the fraction of C4 plants in the community. A statistical model
        is used to predict both forest cover from the GPP for C3 plants and for
        a threshold value indicating closed canopy forest. The ratio of these
        two values is used to indicate the proportion of GPP from trees.

        Note that the GPP units here are in **kilograms** per metre squared per year.

        Args:
            gppc3: The estimated GPP for C3 plants (kg m-2 yr-1).
        """

        prop_trees = (
            self.params.gpp_to_tc_a * np.power(gppc3, self.params.gpp_to_tc_b)
            + self.params.gpp_to_tc_c
        ) / (
            self.params.gpp_to_tc_a
            * np.power(self.params.c3_forest_closure_gpp, self.params.gpp_to_tc_b)
            + self.params.gpp_to_tc_c
        )
        prop_trees = np.clip(prop_trees, 0, 1)

        return prop_trees

    def estimate_isotopic_discrimination(
        self,
        d13CO2: Union[float, np.ndarray],
        Delta13C_C3_alone: Union[float, np.ndarray],
        Delta13C_C4_alone: Union[float, np.ndarray],
    ) -> None:
        r"""Estimate CO2 isotopic discrimination values.

        Creating an instance of {class}`~pyrealm.pmodel.CalcCarbonIsotopes` from
        a {class}`~pyrealm.pmodel.PModel` instance provides estimated total
        annual descrimination against Carbon 13 (:math:`\Delta\ce{^13C}`) for a
        single photosynthetic pathway.

        This method allows predictions from C3 and C4 pathways to be combined to
        calculate the contribution from C3 and C4 plants given the estimated
        fraction of C4 plants. It also calculates the contributions to annual
        stable carbon isotopic composition (:math:`d\ce{^13C}`).

        Four attributes are populated:

        * `Delta13C_C3`: contribution from C3 plants to
          (:math:`\Delta\ce{^13C}`, permil).
        * `Delta13C_C4`: contribution from C4 plants to
          (:math:`\Delta\ce{^13C}`, permil).
        * `d13C_C4`: contribution from C4 plants to (:math:`d\ce{^13C}`,
          permil).
        * `d13C_C3`: contribution from C3 plants to (:math:`d\ce{^13C}`,
          permil).

        Args:
            d13CO2: stable carbon isotopic composition of atmospheric CO2
                (permil)
            Delta13C_C3_alone: annual discrimination against 13C for C3
                plants (permil)
            Delta13C_C4_alone: annual discrimination against 13C for C4
                plants (permil)
        """

        _ = check_input_shapes(
            self.gpp_adv_c4, d13CO2, Delta13C_C3_alone, Delta13C_C4_alone
        )

        self.Delta13C_C3 = Delta13C_C3_alone * (1 - self.frac_c4)
        self.Delta13C_C4 = Delta13C_C4_alone * self.frac_c4

        self.d13C_C3 = (d13CO2 - self.Delta13C_C3) / (1 + self.Delta13C_C3 / 1000)
        self.d13C_C4 = (d13CO2 - self.Delta13C_C4) / (1 + self.Delta13C_C4 / 1000)

    def summarize(self, dp: int = 2) -> None:
        """Print C3C4Competition summary.

        Prints a summary of the calculated values in a C3C4Competition instance
        including the mean, range and number of nan values. This will always
        show fraction of C4 and GPP estaimates and isotopic estimates are shown
        if
        :meth:`~pyrealm.pmodel.C3C4Competition.estimate_isotopic_discrimination`
        has been run.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [
            ("frac_c4", "-"),
            ("gpp_c3_contrib", "gC m-2 yr-1"),
            ("gpp_c4_contrib", "gC m-2 yr-1"),
        ]

        if self.d13C_C3 is not None:
            attrs.extend(
                [
                    ("Delta13C_C3", "permil"),
                    ("Delta13C_C4", "permil"),
                    ("d13C_C3", "permil"),
                    ("d13C_C4", "permil"),
                ]
            )

        summarize_attrs(self, attrs, dp=dp)


# subdaily Pmodel


def memory_effect(values: np.ndarray, alpha: float = 0.067) -> np.ndarray:
    r"""Apply a memory effect to a time series.

    Vcmax and Jmax do not converge instantaneously to acclimated optimal
    values. This function estimates how the actual Vcmax and Jmax track
    a time series of calculated optimal values assuming instant acclimation.

    The estimation uses the paramater `alpha` (:math:`\alpha`) to control
    the speed of convergence of the estimated values (:math:`E`) to the
    calculated optimal values (:math:`O`):

    ::math

        E_{t} = E_{t-1}(1 - \alpha) + O_{t} \alpha

    For :math:`t_{0}`, the first value in the optimal values is used so
    :math:`E_{0} = O_{0}`.

    Args
        values: An equally spaced time series of values
        alpha: The relative weight applied to the most recent observation

    Returns
        An np.ndarray of the same length as `values` with the memory effect
        applied.
    """

    # TODO - NA handling
    # TODO - think about filters here - I'm sure this is a filter which
    #        could generalise to longer memory windows.
    # TODO - need a version that handles time slices for low memory looping
    #        over arrays.

    memory_values = np.empty_like(values, dtype=np.float)
    memory_values[0] = values[0]

    for idx in range(1, len(memory_values)):
        memory_values[idx] = memory_values[idx - 1] * (1 - alpha) + values[idx] * alpha

    return memory_values


def interpolate_rates_forward(
    tk: np.ndarray, ha: float, values: np.ndarray, values_idx: np.ndarray
) -> np.ndarray:
    """Interpolate Jmax and Vcmax forward in time.

    This is a specialised interpolation function used for Jmax and Vcmax. Given
    a time series of temperatures in Kelvin (`tk`) and a set of Jmax25 or
    Vcmax25 values observed at indices (`values_idx`) along that time series,
    this pushes those values along the time series and then rescales to the
    observed temperatures.

    The effect is that the plant 'sets' its response at a given point of the day
    and then maintains that same behaviour until a similar reference time the
    following day.

    Note that the beginning of the sequence will be filled with np.nan values
    unless values_idx[0] = 0.

    Arguments:
        tk: A time series of temperature values (Kelvin)
        ha: An Arrhenius constant.
        values: An array of rates at standard temperature predicted at points along tk.
        values_idx: The indices of tk at which values are predicted.
    """

    v = np.empty_like(tk)
    v[:] = np.nan

    v[values_idx] = values
    v = bn.push(v)

    return v * calc_ftemp_arrh(tk=tk, ha=ha)
