# pylint: disable=C0103
from typing import Optional, Union
from warnings import warn

import numpy as np

from pyrealm.bounds_checker import bounds_checker
from pyrealm.param_classes import PModelParams
from pyrealm.utilities import check_input_shapes, summarize_attrs

# TODO - Note that the typing currently does not enforce the dtype of ndarrays
#        but it looks like the upcoming np.typing module might do this.

def calc_density_h2o(tc: Union[float, np.ndarray],
                     patm: Union[float, np.ndarray],
                     pmodel_params: PModelParams = PModelParams(),
                     safe: bool = True) -> Union[float, np.ndarray]:
    """Calculates the **density of water** as a function of temperature and
    atmospheric pressure, using the Tumlirz Equation and coefficients calculated
    by :cite:`Fisher:1975tm`.

    Parameters:

        tc: air temperature, °C
        patm: atmospheric pressure, Pa
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.
        safe: Prevents the function from estimating density below -30°C, where the
            function behaves poorly

    Other Parameters:

        lambda_: polynomial coefficients of Tumlirz equation (`pmodel_params.fisher_dial_lambda`).
        Po: polynomial coefficients of Tumlirz equation (`pmodel_params.fisher_dial_Po`).
        Vinf: polynomial coefficients of Tumlirz equation (`pmodel_params.fisher_dial_Vinf`).

    Returns:

        Water density as a float in (g cm^-3)

    Examples:

        >>> round(calc_density_h2o(20, 101325), 4)
        998.2056
    """

    # DESIGN NOTE:
    # It doesn't make sense to use this function for tc < 0, but in particular
    # the calculation shows wild numeric instability between -44 and -46 that
    # leads to numerous downstream issues - see the extreme values documentation.
    if safe and np.nanmin(tc) < -30:
        raise RuntimeError('Water density calculations below about -30°C are '
                           'unstable. See argument safe to calc_density_h2o')
        
    # Check input shapes, shape not used
    _ = check_input_shapes(tc, patm)

    # Get powers of tc, including tc^0 = 1 for constant terms
    tc_pow = np.power.outer(tc, np.arange(0, 10))

    # Calculate lambda, (bar cm^3)/g:
    lambda_val = np.sum(np.array(pmodel_params.fisher_dial_lambda) * tc_pow[..., :5], axis=-1)

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
    #     # Calculate density at 1 atm (kg/m^3):
    #     chen_po = np.array([0.99983952, 6.788260e-5 , -9.08659e-6 , 1.022130e-7 ,
    #                         -1.35439e-9 , 1.471150e-11, -1.11663e-13, 5.044070e-16,
    #                         -1.00659e-18])
    #     po = np.sum(np.array(chen_po) * tc_pow[..., :9], axis=-1)
    #
    #     # Calculate bulk modulus at 1 atm (bar):
    #     chen_ko = np.array([19652.17, 148.1830, -2.29995, 0.01281,
    #                         -4.91564e-5, 1.035530e-7])
    #     ko = np.sum(np.array(chen_ko) * tc_pow[..., :6], axis=-1)
    #
    #     # Calculate temperature dependent coefficients:
    #     chen_ca = np.array([3.26138, 5.223e-4, 1.324e-4, -7.655e-7, 8.584e-10])
    #     ca = np.sum(np.array(chen_ca) * tc_pow[..., :5], axis=-1)
    #
    #     chen_cb = np.array([7.2061e-5, -5.8948e-6, 8.69900e-8, -1.0100e-9, 4.3220e-12])
    #     cb = np.sum(np.array(chen_cb) * tc_pow[..., :5], axis=-1)
    #
    #     # Convert atmospheric pressure to bar (1 bar = 100000 Pa)
    #     pbar = 1.0e-5 * patm
    #
    #     rho = (ko + ca * pbar + cb * pbar ** 2.0)
    #     rho /= (ko + ca * pbar + cb * pbar ** 2.0 - pbar)
    #     rho *= 1e3 * po

    return rho


def calc_ftemp_arrh(tk: Union[float, np.ndarray],
                    ha: float,
                    pmodel_params: PModelParams = PModelParams()
                    ) -> Union[float, np.ndarray]:
    r"""Calculates the temperature-scaling factor :math:`f` for enzyme kinetics
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
                  &= exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)\text{, or equivalently}\\
                  &= exp \left( \frac{ H_a}{R} \cdot \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
            \end{align*}
        \]

    Parameters:

        tk: Temperature (in Kelvin)
        ha: Activation energy (in :math:`J \text{mol}^{-1}`)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other Parameters:

        To: a standard reference temperature (:math:`T_0`, `pmodel_params.k_To`)
        R: the universal gas constant (:math:`R`, `pmodel_params.k_R`)

    Returns:

        A float value for :math:`f`

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


def calc_ftemp_inst_rd(tc: Union[float, np.ndarray],
                       pmodel_params: PModelParams = PModelParams()
                       ) -> Union[float, np.ndarray]:
    """Calculates the **temperature-scaling factor for dark respiration**
    at a given temperature (``tc``, :math:`T` in °C), relative to the standard
    reference temperature :math:`T_o` (:cite:`Heskel:2016fg`).

    .. math::

            fr = exp( b (T_o - T) -  c ( T_o^2 - T^2 ))

    Parameters:

        tc: Temperature (degrees Celsius)

    Other parameters:

        To: standard reference temperature (:math:`T_o`, `pmodel_params.k_To`)
        b: empirically derived global mean coefficient (:math:`b`, Table 1, ::cite:`Heskel:2016fg`)
        c: empirically derived global mean coefficient (:math:`c`, Table 1, ::cite:`Heskel:2016fg`)


    Returns:

        A float value for :math:`fr`

    Examples:

        >>> # Relative percentage instantaneous change in Rd going from 10 to 25 degrees
        >>> val = (calc_ftemp_inst_rd(25) / calc_ftemp_inst_rd(10) - 1) * 100
        >>> round(val, 4)
        250.9593
    """

    return np.exp(pmodel_params.heskel_b * (tc - pmodel_params.k_To) -
                  pmodel_params.heskel_c * (tc ** 2 - pmodel_params.k_To ** 2))


def calc_ftemp_inst_vcmax(tc: Union[float, np.ndarray],
                          pmodel_params: PModelParams = PModelParams()
                          ) -> Union[float, np.ndarray]:
    r"""This function calculates the **temperature-scaling factor :math:`f` of
    the instantaneous temperature response of :math:`V_{cmax}`** given the
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

    Parameters:

        tc:  temperature, or in general the temperature relevant for
            photosynthesis (°C)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other parameters:

        Ha: activation energy (:math:`H_a`, `pmodel_params.kattge_knorr_Ha`)
        Hd: deactivation energy (:math:`H_d`, `pmodel_params.kattge_knorr_Hd`)
        To: standard reference temperature expressed in Kelvin (`T_0`, `pmodel_params.k_To`)
        R: the universal gas constant (:math:`R`,`pmodel_params.k_R`)
        a: intercept of the entropy factor(:math:`a`, `pmodel_params.kattge_knorr_a_ent`)
        b: slope of the entropy factor (:math:`b`, `pmodel_params.kattge_knorr_b_ent`)

    Returns: A float value for :math:`f`

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
    fvb = ((1 + np.exp((tkref * dent - pmodel_params.kattge_knorr_Hd) /
                       (pmodel_params.k_R * tkref))) /
           (1 + np.exp((tk * dent - pmodel_params.kattge_knorr_Hd) /
                       (pmodel_params.k_R * tk))))

    return fva * fvb

# TODO - update unpublished reference to:
#  Cai, W., and Prentice, I. C.: Recent trends in gross primary production 
#        and their drivers: analysis and modelling at flux-site and global scales,
#        Environ. Res. Lett. 15 124050 https://doi.org/10.1088/1748-9326/abc64e, 2020

def calc_ftemp_kphio(tc: Union[float, np.ndarray],
                     c4: bool = False,
                     pmodel_params: PModelParams = PModelParams()
                     ) -> Union[float, np.ndarray]:
    r"""Calculates the **temperature dependence of the quantum yield
    efficiency**, as a quadratic function of temperature (:math:`T`). The values
    of the coefficients depend on whether C3 or C4 photosynthesis is being
    modelled

    .. math::

        \phi(T) = a + b T - c T^2

    The factor :math:`\phi(T)` is to be multiplied with leaf absorptance and the
    fraction of absorbed light that reaches photosystem II. In the P-model these
    additional factors are lumped into a single apparent quantum yield
    efficiency parameter (argument `kphio` to the class :class:`~pyrealm.pmodel.PModel`).

    Parameters:

        tc: Temperature, relevant for photosynthesis (°C)
        c4: Boolean specifying whether fitted temperature response for C4 plants
            is used. Defaults to \code{FALSE}.
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other parameters:

        C3: the parameters (:math:`a,b,c`, `pmodel_params.kphio_C3`) are taken from the
            temperature dependence of the maximum quantum yield of photosystem
            II in light-adapted tobacco leaves determined by :cite:`Bernacchi:2003dc`.
        C4: the parameters (:math:`a,b,c`, `pmodel_params.kphio_C4`) are taken from unpublished
            work.

    Returns:

        A float value for :math:`\phi(T)`

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

    ftemp = coef[0] + coef[1] * tc + coef[2] * tc ** 2
    ftemp = np.clip(ftemp, 0.0, None)
    
    return ftemp


def calc_gammastar(tc: Union[float, np.ndarray],
                   patm: Union[float, np.ndarray],
                   pmodel_params: PModelParams = PModelParams()
                   ) -> Union[float, np.ndarray]:
    r"""Calculates the photorespiratory **CO2 compensation point** in absence of
    dark respiration (:math:`\Gamma^{*}`, ::cite:`Farquhar:1980ft`) as:

    .. math::

        \Gamma^{*} = \Gamma^{*}_{0} \cdot \frac{p}{p_0} \cdot f(T, H_a)

    where :math:`f(T, H_a)` modifies the activation energy to the the local
    temperature following an Arrhenius-type temperature response function
    implemented in :func:`calc_ftemp_arrh`.

    Parameters:

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

        A float value for :math:`\Gamma^{*}` (in Pa)

    Examples:

        >>> # CO2 compensation point at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa) >>> round(calc_gammastar(20, 101325), 5)
        3.33925
    """

    # check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    return (pmodel_params.bernacchi_gs25_0 * patm / pmodel_params.k_Po *
            calc_ftemp_arrh((tc + pmodel_params.k_CtoK), ha=pmodel_params.bernacchi_dha))


def calc_ns_star(tc: Union[float, np.ndarray],
                 patm: Union[float, np.ndarray],
                 pmodel_params: PModelParams = PModelParams()
                 ) -> Union[float, np.ndarray]:

    r"""Calculates the relative viscosity of water (:math:`\eta^*`), given the
    standard temperature and pressure, using :func:`~pyrealm.pmodel.calc_viscosity_h20`
    (:math:`v(t,p)`) as:

    .. math::

        \eta^* = \frac{v(t,p)}{v(t_0,p_0)}

    Parameters:

        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other parameters:

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

    visc_env = calc_viscosity_h2o(tc, patm)
    visc_std = calc_viscosity_h2o(pmodel_params.k_To, pmodel_params.k_Po)

    return visc_env / visc_std


def calc_kmm(tc: Union[float, np.ndarray],
             patm: Union[float, np.ndarray],
             pmodel_params: PModelParams = PModelParams()
             ) -> Union[float, np.ndarray]:
    r"""Calculates the **Michaelis Menten coefficient of Rubisco-limited
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

    Parameters:

        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other parameters:

        hac: activation energy for :math:`\ce{CO2}` (:math:`H_{kc}`, `pmodel_params.bernacchi_dhac`)
        hao:  activation energy for :math:`\ce{O2}` (:math:`\Delta H_{ko}`, `pmodel_params.bernacchi_dhao`)
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

    kc = pmodel_params.bernacchi_kc25 * calc_ftemp_arrh(tk, ha=pmodel_params.bernacchi_dhac)
    ko = pmodel_params.bernacchi_ko25 * calc_ftemp_arrh(tk, ha=pmodel_params.bernacchi_dhao)

    # O2 partial pressure
    po = pmodel_params.k_co * 1e-6 * patm

    return kc * (1.0 + po/ko)


def calc_kp_c4(tc: Union[float, np.ndarray],
             patm: Union[float, np.ndarray],
             pmodel_params: PModelParams = PModelParams()
             ) -> Union[float, np.ndarray]:
    r"""Calculates the **Michaelis Menten coefficient of phosphoenolpyruvate
    carboxylase (PEPc)** (:math:`K`, ::cite:`Boyd:2015ft`) as a function of
    temperature (:math:`T`) and atmospheric pressure (:math:`p`) as:

    Parameters:

        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other parameters:

        hac: activation energy for :math:`\ce{CO2}` (:math:`H_{kc}`, `pmodel_params.boyd_dhac_c4`)
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

    kc = pmodel_params.boyd_kp25_c4 * calc_ftemp_arrh(tk, ha=pmodel_params.boyd_dhac_c4)

    return kc
    
    
def calc_soilmstress(soilm: Union[float, np.ndarray],
                     meanalpha: Union[float, np.ndarray] = 1.0,
                     pmodel_params: PModelParams = PModelParams()
                     ) -> Union[float, np.ndarray]:
    r"""Calculates an **empirical soil moisture stress factor**  (:math:`\beta`,
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
                    q(\theta_0 - \theta^{*})^2 + 1,  & \theta_0 < m_s <= \theta^{*} \\
                    1, &  \theta^{*} < m_s,
                \end{cases}
        \]

    where :math:`q` is an aridity sensitivity parameter setting the stress
    factor at :math:`\theta_0`:

    .. math:: q=(1 - (a + b \bar{\alpha}))/(\theta^{*} - \theta_{0})^2

    Default parameters are as described in :cite:`Stocker:2020dh`.

    Parameters:

        soilm: Relative soil moisture as a fraction of field capacity
            (unitless). Defaults to 1.0 (no soil moisture stress).
        meanalpha: Local annual mean ratio of actual over potential
            evapotranspiration, measure for average aridity. Defaults to 1.0.
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other parameters:

        theta0: lower bound of soil moisture (:math:`\theta_0`, `pmodel_params.soilmstress_theta0`).
        thetastar: upper bound of soil moisture (:math:`\theta^{*}`, `pmodel_params.soilmstress_thetastar`).
        a: aridity parameter (:math:`a`, `pmodel_params.soilmstress_a`).
        b: aridity parameter (:math:`b`, `pmodel_params.soilmstress_b`).

    Returns:

        A numeric value for :math:`\beta`

    Examples:

        >>> # Relative reduction (%) in GPP due to soil moisture stress at
        >>> # relative soil water content ('soilm') of 0.2:
        >>> round((calc_soilmstress(0.2) - 1) * 100, 5)
        -14.0
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(soilm, meanalpha)

    # Calculate outstress
    y0 = (pmodel_params.soilmstress_a + pmodel_params.soilmstress_b * meanalpha)
    beta = (1.0 - y0) / (pmodel_params.soilmstress_theta0 - pmodel_params.soilmstress_thetastar) ** 2
    outstress = 1.0 - beta * (soilm - pmodel_params.soilmstress_thetastar) ** 2

    # Filter wrt to thetastar
    outstress = np.where(soilm <= pmodel_params.soilmstress_thetastar, outstress, 1.0)

    # Clip
    outstress = np.clip(outstress, 0.0, 1.0)

    return outstress


def calc_viscosity_h2o(tc: Union[float, np.ndarray],
                       patm: Union[float, np.ndarray],
                       pmodel_params: PModelParams = PModelParams()
                       ) -> Union[float, np.ndarray]:
    r"""Calculates the **viscosity of water** (:math:`\eta`) as a function of
    temperature and atmospheric pressure (::cite:`Huber:2009fy`).

    Parameters:

        tc: air temperature (°C)
        patm: atmospheric pressure (Pa)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Returns:

        A float giving the viscosity of water (mu, Pa s)

    Examples:

        >>> # Density of water at 20 degrees C and standard atmospheric pressure:
        >>> round(calc_viscosity_h2o(20, 101325), 7)
        0.0010016
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    # Get the density of water, kg/m^3
    rho = calc_density_h2o(tc, patm, pmodel_params=pmodel_params)

    # Calculate dimensionless parameters:
    tbar = (tc + pmodel_params.k_CtoK) / pmodel_params.huber_tk_ast
    rbar = rho / pmodel_params.huber_rho_ast

    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    tbar_pow = np.power.outer(tbar, np.arange(0, 4))
    mu0 = (1e2 * np.sqrt(tbar)) / np.sum(np.array(pmodel_params.huber_H_i) / tbar_pow, axis=-1)

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


def calc_patm(elv: Union[float, np.ndarray],
              pmodel_params: PModelParams = PModelParams()
              ) -> Union[float, np.ndarray]:
    r"""Calculates **atmospheric pressure** as a function of elevation with reference to
    the standard atmosphere.  The elevation-dependence of atmospheric pressure
    is computed by assuming a linear decrease in temperature with elevation and
    a mean adiabatic lapse rate (Eqn 3, ::cite:`BerberanSantos:2009bk`):

    .. math::

        p(z) = p_0 ( 1 - L z / K_0) ^{ G M / (R L) },

    Parameters:

        elv: Elevation above sea-level (:math:`z`, metres above sea level.)
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.

    Other parameters:

        G: gravity constant (:math:`g`, `pmodel_params.k_G`)
        Po: standard atmospheric pressure at sea level (:math:`p_0`, `pmodel_params.k_Po`)
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

    return (pmodel_params.k_Po * (1.0 - pmodel_params.k_L * elv / kto) **
            (pmodel_params.k_G * pmodel_params.k_Ma /
             (pmodel_params.k_R * pmodel_params.k_L)))


def calc_co2_to_ca(co2: Union[float, np.ndarray],
                   patm: Union[float, np.ndarray],
                   ) -> Union[float, np.ndarray]:
    r"""Converts ambient :math:`\ce{CO2}` (:math:`c_a`) in part per million to
    Pascals, accounting for atmospheric pressure.

    Parameters:
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

    """Create a PModelEnvironment instance using the input parameters.

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

    Parameters:

        tc: Temperature, relevant for photosynthesis (°C)
        vpd: Vapour pressure deficit (Pa)
        theta: Volumetric soil moisture (m3/m3)
        co2: Atmospheric CO2 concentration (ppm)
        patm: Atmospheric pressure (Pa).
        pmodel_params: An instance of :class:`~pyrealm.param_classes.PModelParams`.
        d13CO2: Atmospheric d13CO2 (‰)
        D14CO2: Atmospheric D14CO2 (‰)
    """

    def __init__(self,
                 tc: Union[float, np.ndarray],
                 vpd: Union[float, np.ndarray],
                 theta: Union[float, np.ndarray],
                 co2: Union[float, np.ndarray],
                 patm: Union[float, np.ndarray],
                 d13CO2: Union[float, np.ndarray],
                 D14CO2: Union[float, np.ndarray],
                 pmodel_params: PModelParams = PModelParams()):

        self.shape = check_input_shapes(tc, vpd, co2, patm)

        # Validate and store the forcing variables
        self.tc = bounds_checker(tc, -25, 80, '[]', 'tc', '°C')
        self.vpd = bounds_checker(vpd, 0, 10000, '[]', 'vpd', 'Pa')
        self.theta = bounds_checker(theta, 0, 0.8, '[]', 'theta', 'm3/m3')
        self.co2 = bounds_checker(co2, 0, 1000, '[]', 'co2', 'ppm')
        self.patm = bounds_checker(patm, 30000, 110000, '[]', 'patm', 'Pa')
        self.d13CO2 = bounds_checker(d13CO2, -10, -6, '[]', 'd13CO2', '‰')
        self.D14CO2 = bounds_checker(D14CO2, -25, 800, '[]', 'D14CO2', '‰')

        # Guard against calc_density issues
        if np.nanmin(self.tc) < -25:
            raise ValueError('Cannot calculate P Model predictions for values below -25°C. See calc_density_h2o.')

        # Guard against negative VPD issues
        if np.nanmin(self.vpd) < 0:
            raise ValueError('Negative VPD values will lead to missing data - clip to zero or explicitly set to np.nan')

        # ambient CO2 partial pressure (Pa)
        self.ca = calc_co2_to_ca(self.co2, self.patm)

        # photorespiratory compensation point - Gamma-star (Pa)
        self.gammastar = calc_gammastar(tc, patm, pmodel_params=pmodel_params)

        # Michaelis-Menten coef. (Pa)
        self.kmm = calc_kmm(tc, patm, pmodel_params=pmodel_params)

        # Michaelis-Menten coef. C4 plants (Pa)
        self.kp_c4 = calc_kp_c4(tc, patm, pmodel_params=pmodel_params)
        
        # viscosity correction factor relative to standards
        self.ns_star = calc_ns_star(tc, patm, pmodel_params=pmodel_params)  # (unitless)

        # Store parameters
        self.pmodel_params = pmodel_params

    def __repr__(self):

        # DESIGN NOTE: This is deliberately extremely terse. It could contain
        # a bunch of info on the environment but that would be quite spammy
        # on screen. Having a specific summary method that provides that info
        # is more user friendly.

        return f"PModelEnvironment(shape={self.shape})"

    def summarize(self, dp=2):
        """Prints a summary of the input and photosynthetic attributes in a
        instance of a PModelEnvironment including the mean, range and number
        of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.

        Returns:
            None
        """

        attrs = [('tc', '°C'), ('vpd', 'Pa'), ('theta', 'm3/m3'), ('co2', 'ppm'),
                 ('patm', 'Pa'), ('ca', 'Pa'), ('d13CO2', '‰'), ('D14CO2', '‰'),
                 ('gammastar', 'Pa'), ('kmm', 'Pa'), ('kp_c4', 'Pa'), ('ns_star', '-')]
        summarize_attrs(self, attrs, dp=dp)


class PModel:

    r"""Fits the P Model to a given set of environmental and photosynthetic
    parameters. The calculated attributes of the class are described below. An
    extended description with typical use cases is given in
    :any:`pmodel_overview` but the basic flow of the model is:

    1. Estimate :math:`\ce{CO2}` limitation factors and optimal internal to
       ambient :math:`\ce{CO2}` partial pressure ratios (:math:`\chi`), using
       :class:`~pyrealm.pmodel.CalcOptimalChi`.
    2. Estimate photosynthetic efficiencies including water use efficiency,
       light use efficiency (LUE) and maximum carboxylation rate
       (:math:`V_{cmax}`) using :class:`~pyrealm.pmodel.CalcLUEVcmax`.
    3. Optionally, estimate productivity measures including GPP by supplying
       FAPAR and PPFD using the
       :meth:`~pyrealm.pmodel.PModel.estimate_productivity` method.

    **Corollary prediction details**

    These calculations use two additional functions:

    * the instantaneous temperature response of :math:`V_{cmax}`
      (:math:`fv(t)`), implemented in
      :func:`~pyrealm.pmodel.calc_ftemp_inst_vcmax`, and
    * the instantaneous temperature response of dark respiration :math:`V_{d}`
      (:math:`fr(t)`), implemented in
      :func:`~pyrealm.pmodel.calc_ftemp_inst_rd`.

    The predictions are then:

    * Intrinsic water use efficiency (iWUE,
      :math:`\mu\mathrm{mol}\;\mathrm{mol}^{-1}`), calculated as :math:`( 5/8 *
      (c_a - c_i)) / P`, where `c_a` and `c_i` are measured in Pa and :math:`P`
      is atmospheric pressure in megapascals. This is equivalent to :math:`(c_a
      - c_i)/1.6` when `c_a` and `c_i` are expressed as parts per million.

    * Maximum carboxylation capacity (mol C m-2) normalised to the standard
      temperature as: :math:`V_{cmax25} = V_{cmax}  / fv(t)`

    * Dark respiration, calculated as:

        .. math::

            R_d = b_0 \frac{fr(t)}{fv(t)} V_{cmax}

        following :cite:`Atkin:2015hk` (:math:`b_0` is set in
        :attr:`~pyrealm.pmodel_params.atkin_rd_to_vcmax`)

    * Stomatal conductance (:math:`g_s`), calculated as:

        .. math::

            g_s = \frac{LUE}{M_C}\frac{1}{c_a - c_i}

        When C4 photosynthesis is being used, the true partial pressure of CO2
        in the substomatal cavities (:math:`c_i`) is used following the
        calculation of :math:`\chi` using
        :attr:`~pyrealm.param_classes.PModelParams.beta_cost_ratio_c4`

    * The maximum rate of Rubsico regeneration at the growth temperature
      (:math:`J_{max}`) per unit irradiance is calculated as:

        .. math::

            J_{max} = \frac{4 \phi_0 I_{abs}}{\sqrt{\left(\frac{1}
            {\left(\frac{V_{cmax}(c_i - 2 \Gamma^*)} {\phi_0 I_{abs}(c_i +
            k_{mm})}\right)}\right)^2 - 1}}

    Parameters:

        env: An instance of :class:`~pyrealm.pmodel.PModelEnvironment`. kphio:
        (Optional) Apparent quantum yield efficiency (unitless). rootzonestress:
        (Optional, default=None) An experimental option
            for providing a root zone water stress factor. This is not
            compatible with the soilmstress approach.
        soilmstress: (Optional, default=None) A soil moisture stress factor
            calculated using :func:`~pyrealm.pmodel.calc_soilmstress`.
        c4: (Optional, default=False) Selects the C3 or C4 photosynthetic
            pathway. 
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
        iwue: Intrinsic water use efficiency (iWUE, µmol mol-1) 
        lue: Light use efficiency (LUE, g C mol-1)

        After :meth:`~pyrealm.pmodel.estimate_productivity` has been run,
        the following attributes are also populated. See the documentation
        for :meth:`~pyrealm.pmodel.estimate_productivity` for notes on the 
        units of these attributes

        gpp: Gross primary productivity (gC area time)
        rd: Dark respiration (mol C area time)
        vcmax: Maximum rate of carboxylation (mol C area time)
        vcmax25: Maximum rate of carboxylation at standard temperature (mol C area time)
        jmax: Maximum rate of electron transport (mol C area time)
        gs: Stomatal conductance (mol C area time)

    Examples:

        >>> env = PModelEnvironment(tc=20, vpd=1000, co2=400, patm=101325.0, theta=0.4,d13CO2= -8.4, D14CO2 = 19.2)
        >>> mod_c3 = PModel(env)
        >>> # Key variables from pmodel
        >>> round(mod_c3.optchi.ci, 5)
        29.18551
        >>> round(mod_c3.optchi.chi, 5)
        0.7201
        >>> mod_c3.estimate_productivity(fapar=1, ppfd=300)
        >>> round(mod_c3.gpp, 5)
        77.99435
        >>> mod_c4 = PModel(env, c4=True, method_jmaxlim='none')
        >>> # Key variables from PModel
        >>> round(mod_c4.optchi.ci, 5)
        15.88362
        >>> round(mod_c4.optchi.chi, 5)
        0.3919
        >>> mod_c4.estimate_productivity(fapar=1, ppfd=300)
        >>> round(mod_c4.gpp, 5)
        103.25886
    """

    def __init__(self,
                 env: PModelEnvironment,
                 rootzonestress: Optional[Union[float, np.ndarray]] = None,
                 soilmstress: Optional[Union[float, np.ndarray]] = None,
                 kphio: Optional[float] = None,
                 do_ftemp_kphio: bool = True,
                 c4: bool = False,
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

        if soilmstress is not None and rootzonestress is not None:
            raise AttributeError("Soilmstress and rootzonestress are alternative "
                                 "approaches to soil moisture effects. Do not use both.")

        if soilmstress is None:
            soilmstress = 1.0
            self.do_soilmstress = False
        else:
            self.do_soilmstress = True

        if rootzonestress is None:
            rootzonestress = 1.0
            self.do_rootzonestress = False
        else:
            self.do_rootzonestress = True

        # kphio defaults:
        self.do_ftemp_kphio = do_ftemp_kphio
        if kphio is None:
            if not self.do_ftemp_kphio:
                self.kphio = 0.049977
            elif self.do_soilmstress:
                self.kphio = 0.087182
            else:
                self.kphio = 0.081785
        else:
            self.kphio = kphio

        # -----------------------------------------------------------------------
        # Temperature dependence of quantum yield efficiency
        # -----------------------------------------------------------------------
        # 'do_ftemp_kphio' is not actually a stress function, but is the temperature-
        # dependency of the quantum yield efficiency after Bernacchi et al., 2003

        if self.do_ftemp_kphio:
            self.ftemp_kphio = calc_ftemp_kphio(env.tc, c4,
                                                pmodel_params=env.pmodel_params)
        else:
            self.ftemp_kphio = 1.0

        # -----------------------------------------------------------------------
        # Optimal ci
        # The heart of the P-model: calculate ci:ca ratio (chi) and additional terms
        # -----------------------------------------------------------------------
        self.c4 = c4

        if self.c4:
            method_optci = "c4"
        else:
            method_optci = "c3"

        self.optchi = CalcOptimalChi(env=env, method=method_optci,
                                     rootzonestress=rootzonestress,
                                     pmodel_params=env.pmodel_params)

        # -----------------------------------------------------------------------
        # Carbon Isotopes
        # -----------------------------------------------------------------------
        self.c4 = c4

        if self.c4:
            method_discr = "c4"
        else:
            method_discr = "c3"

        self.delta = CalcCarbonIsotopes(self.optchi, env.gammastar, env.ca, env.tc,
                                        env.d13CO2,env.D14CO2,method=method_discr,
                                        pmodel_params=env.pmodel_params)

        # -----------------------------------------------------------------------
        # Vcmax and light use efficiency
        # -----------------------------------------------------------------------

        self.method_jmaxlim = method_jmaxlim
        lue_vcmax = CalcLUEVcmax(self.optchi, self.kphio,
                                 self.ftemp_kphio, soilmstress,
                                 method=method_jmaxlim,
                                 pmodel_params=env.pmodel_params)

        # -----------------------------------------------------------------------
        # Store the two efficiency predictions
        # -----------------------------------------------------------------------

        # intrinsic water use efficiency (in µmol mol-1). The rpmodel reports this
        # in Pascals, but more commonly reported in µmol mol-1. The standard equation
        # (ca - ci) / 1.6 expects inputs in ppm, so the pascal versions are back
        # converted here.
        self.iwue = (5/8 * (env.ca - self.optchi.ci)) / (1e-6 * self.env.patm)
        self.lue = lue_vcmax.lue
        self.vcmax_unit_iabs = lue_vcmax.vcmax

        # -----------------------------------------------------------------------
        # Define attributes populated by estimate_productivity method
        # -----------------------------------------------------------------------
        self._vcmax = None
        self._vcmax25 = None
        self._rd = None
        self._jmax = None
        self._gpp = None
        self._gs = None

    @property
    def gpp(self) -> Union[float, np.ndarray]:
        """Cannot return GPP if estimate_productivity has not been run, do
           not return None silently"""
        if self._gpp is None:
            raise RuntimeError('GPP not calculated: use estimate_productivity')

        return self._gpp

    @property
    def vcmax(self) -> Union[float, np.ndarray]:
        """Cannot return V_cmax if estimate_productivity has not been run, do
           not return None silently"""
        if self._vcmax is None:
            raise RuntimeError('vcmax not calculated: use estimate_productivity')

        return self._vcmax

    @property
    def vcmax25(self) -> Union[float, np.ndarray]:
        """Cannot return V_cmax25 if estimate_productivity has not been run, do
           not return None silently"""
        if self._vcmax25 is None:
            raise RuntimeError('vcmax25 not calculated: use estimate_productivity')

        return self._vcmax25

    @property
    def rd(self) -> Union[float, np.ndarray]:
        """Cannot return RD if estimate_productivity has not been run, do
           not return None silently"""
        if self._rd is None:
            raise RuntimeError('RD not calculated: use estimate_productivity')

        return self._rd

    @property
    def jmax(self) -> Union[float, np.ndarray]:
        """Cannot return Jmax if estimate_productivity has not been run, do
           not return None silently"""
        if self._jmax is None:
            raise RuntimeError('Jmax not calculated: use estimate_productivity')

        return self._jmax

    @property
    def gs(self) -> Union[float, np.ndarray]:
        """Cannot return gs if estimate_productivity has not been run, do
           not return None silently"""
        if self._gs is None:
            raise RuntimeError('GS not calculated: use estimate_productivity')

        return self._gs

    def estimate_productivity(self,
                              fapar: Union[float, np.ndarray] = 1,
                              ppfd: Union[float, np.ndarray] = 1):
        r""" This function takes the light use efficiency and Vcmax per unit
        absorbed irradiance and populates the PModel instance with estimates
        of the following:

            * gpp: Gross primary productivity
            * rd: Dark respiration
            * vcmax: Maximum rate of carboxylation
            * vcmax25: Maximum rate of carboxylation at standard temperature
            * jmax: Maximum rate of electron transport.
            * gs: Stomatal conductance

        The functions finds the total absorbed irradiance (:math:`I_{abs}`) as
        the product of the photosynthetic photon flux density (`ppfd`) and the
        fraction of absorbed photosynthetically active radiation (`fapar`).

        Note that the units of PPFD determine the units of outputs: if PPFD is
        in :math:`\text{mol} m^{-2} \text{month}^{-1}`, then output values are
        scaled per square metre per month.

        The default values of ``ppfd`` and ``fapar`` provide estimates of the
        variables above per unit absorbed irradiance.

        Args:
            fapar: the fraction of absorbed photosynthetically active radiation
            ppfd: photosynthetic photon flux density
        """

        # Check input shapes against each other and an existing calculated value
        _ = check_input_shapes(ppfd, fapar, self.lue)
        
        # Calculate Iabs 
        iabs = fapar * ppfd
        
        # GPP
        self._gpp = self.lue * iabs

        # V_cmax
        self._vcmax = self.vcmax_unit_iabs * iabs

        # V_cmax25 (vcmax normalized to pmodel_params.k_To)
        ftemp25_inst_vcmax = calc_ftemp_inst_vcmax(self.env.tc,
                                                   pmodel_params=self.pmodel_params)
        self._vcmax25 = self.vcmax / ftemp25_inst_vcmax

        # Dark respiration at growth temperature
        ftemp_inst_rd = calc_ftemp_inst_rd(self.env.tc,
                                           pmodel_params=self.pmodel_params)
        self._rd = (self.pmodel_params.atkin_rd_to_vcmax *
                    (ftemp_inst_rd / ftemp25_inst_vcmax) * self.vcmax)

        # Jmax using again A_J = A_C
        fact_jmaxlim = (self.vcmax * (self.optchi.ci + 2.0 * self.env.gammastar) /
                        (self.kphio * iabs * (self.optchi.ci + self.env.kmm)))
        
        # The equation Jmax = 4 * phio * I_abs  / sqrt((1 / Jf) ^ 2 -1)
        # has the domain [-1, 0) and (0, 1], so need to be careful about
        # what values of fact_jmaxlim are allowed.
        # Guard against negative values getting into sqrt
        jmaxlim_step1 = (1.0 / fact_jmaxlim) ** 2 - 1.0
        jmax = np.empty_like(fact_jmaxlim)
        mask = jmaxlim_step1 > 0
        
        # Iabs might be a scalar or an array - if an array, it should be of the
        # same shape as jmaxlim_step1 and can use the same mask
        iabs_mask = iabs if isinstance(iabs, (float, int)) else iabs[mask]
        jmax[mask] = 4.0 * self.kphio * iabs_mask / np.sqrt(jmaxlim_step1[mask])
        jmax[~ mask] = np.nan

        # Revert to scalar if needed and store
        self._jmax = jmax.item() if np.ndim(jmax) == 0 else jmax

        # AJ and AC
        a_j = (self.kphio * iabs * (self.optchi.ci - self.env.gammastar) / 
                (self.optchi.ci + 2 * self.env.gammastar) *  fact_jmaxlim)
        a_c = (self.vcmax * (self.optchi.ci - self.env.gammastar)/ 
                (self.optchi.ci + self.env.kmm))
        assim = np.minimum(a_j, a_c)

        if not np.allclose(assim, self._gpp / self.pmodel_params.k_c_molmass, equal_nan=True):
            warn('Assimilation and GPP are not identical')

        # Stomatal conductance
        self._gs = assim / (self.env.ca - self.optchi.ci)

    def __repr__(self):
        if self.do_soilmstress:
            stress = 'Soil moisture'
        elif self.do_rootzonestress:
            stress = 'Root zone'
        else:
            stress = 'None'
        return (f"PModel("
                f"shape={self.shape}, "
                f"kphio={self.kphio}, "
                f"ftemp_kphio={self.do_ftemp_kphio}, "
                f"c4={self.c4}, "
                f"Jmax_method={self.method_jmaxlim}, "
                f"Water stress={stress})")

    def summarize(self, dp=2):
        """Prints a summary of the calculated values in a PModel instance
        including the mean, range and number of nan values. This will always
        show efficiency variables (LUE and IWUE) and productivity estimates are
        shown if :meth:`~pyrealm.pmodel.PModel.calculate_productivity` has been
        run.

        Args:
            dp: The number of decimal places used in rounding summary stats.

        Returns:
            None
        """

        attrs = [('lue', 'g C mol-1'), 
                 ('iwue', 'µmol mol-1')]

        if self._gpp:
            attrs.extend([('gpp', 'gC area time †'),
                          ('vcmax', 'mol C area time †'),
                          ('vcmax25', 'mol C area time †'), 
                          ('rd', 'mol C area time †'),
                          ('gs', 'mol C area time †'),
                          ('jmax', 'mol C area time †')])

        summarize_attrs(self, attrs, dp=dp)

        if self._gpp:
            print('\n†: The units of all these variables follow the area and\n'
                    '   time scaling of the PPFD data used.')


class CalcOptimalChi:
    r"""Calculate the optimal :math:`\chi` and :math:`\ce{CO2}` limitation
    factors. In more details, the values are:

    - The optimal ratio of leaf internal to ambient :math:`\ce{CO2}` partial
      pressure (:math:`\chi = c_i/c_a`).
    - The :math:`\ce{CO2}` limitation term for light-limited
      assimilation (:math:`m_j`).
    - The :math:`\ce{CO2}` limitation term for Rubisco-limited
      assimilation  (:math:`m_c`).

    The value for :math:`\chi` is calculated using the parameter :math:`\beta`,
    which differs between C3 and C4 plants. :cite:`Stocker:2020dh` estimated
    $\beta = 146$ for C3 plants, and this is defined as `beta_unit_cost_c3` in
    :class:`~pyrealm.param_classes.PModelParams`. For C4 plants, the default
    value used is $\beta = 146 /  9 \approx 16.222$, defined as
    `beta_unit_cost_c4` in :class:`~pyrealm.param_classes.PModelParams`.

    The chosen method is automatically used to estimate these values when an
    instance is created.

    Attributes:

        env (PModelEnvironment): An instance of PModelEnvironment providing
            the photosynthetic environment for the model.
        method (str): one of ``c4`` or ``c3``
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

    Examples:

        >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000, theta=0.4,d13CO2= -8.4, D14CO2 = 19.2)
        >>> vals = CalcOptimalChi(env=env)
        >>> round(vals.chi, 5)
        0.7201
        >>> round(vals.mc, 5)
        0.34331
        >>> round(vals.mj, 5)
        0.72067
        >>> round(vals.mjoc, 5)
        2.09917
        >>> # The first c4 method estimates chi but sets the others at 1.
        >>> # The second c4 method estimates chi, assumed gammastar to be negligible so that mj equal to 1.
        >>> vals = CalcOptimalChi(env=env, method='c4')
        >>> round(vals.chi, 5)
        0.3919
        >>> round(vals.mc, 5)
        0.25626
        >>> round(vals.mj, 5)
        1.0
        >>> round(vals.mjoc, 5)
        3.90232
    """

    # TODO - move chi calc into __init__? Shared between the two methods

    def __init__(self,
                 env: PModelEnvironment,
                 rootzonestress: Union[float, np.ndarray] = 1.0,
                 method: str = 'c3',
                 pmodel_params: PModelParams = PModelParams()
                 ):

        # Check rootzone stress is broadcastable to the environment
        self.shape = check_input_shapes(env.ca, rootzonestress)

        # set attribute defaults
        self.xi = None
        self.chi = None
        self.ci = None
        self.mc = None
        self.mj = None
        self.mjoc = None

        # Identify and run the selected method
        self.pmodel_params = pmodel_params
        self.method = method
        all_methods = {'c3': self.c3, 'c4': self.c4}

        if self.method in all_methods:
            this_method = all_methods[self.method]
            this_method(env = env, rootzonestress=rootzonestress)
        else:
            raise ValueError(f"CalcOptimalChi: method argument '{method}' invalid.")

        # Calculate internal CO2 partial pressure
        self.ci = self.chi * env.ca

    def __repr__(self):

        return f"CalcOptimalChi(shape={self.shape}, method={self.method})"

    def c4(self, env, rootzonestress):
        r"""Optimal :math:`\chi` is calculated following Equation 8 in
        :cite:`Prentice:2014bc` (see :meth:`~pyrealm.pmodel.CalcOptimalChi.c3`),
        but using a C4 specific estimate of the unit cost ratio :math:`\beta`,
        specified in :meth:`~pyrealm.param_classes.PModelParams.beta_cost_ratio_c4`.

        Method 1: This method then simply sets :math:`m_j = m_c = m_{joc} = 1.0` to capture the
        boosted :math:`\ce{CO2}` concentrations at the chloropolast in C4
        photosynthesis.
        Method 2: This method sets :math:`m_j = 1.0` because gammastar is negligible (equal to 0)
        in C4 photosynthesis.
        """

        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        # version 1: main version
        #self.xi = np.sqrt((self.pmodel_params.beta_cost_ratio_c4 * rootzonestress *
        #                    (env.kmm + env.gammastar))
        #                  / (1.6 * env.ns_star))

        #self.chi = (env.gammastar / env.ca + (1.0 - env.gammastar / env.ca) * self.xi
        #            / (self.xi + np.sqrt(env.vpd)))
       
       
        # version 2: simple formula assuming that photorespiration is negligible
        self.xi = np.sqrt((self.pmodel_params.beta_cost_ratio_c4 * rootzonestress *env.kmm)
                     / (1.6 * env.ns_star))
        
        self.chi = self.xi /(self.xi + np.sqrt(env.vpd))
                
        # version 3 as in Scott & Smith (2022)
        #beta_c4 = 166
        #self.xi = np.sqrt((beta_c4 *env.kp_c4) / (1.6 * env.ns_star))
                     
        #self.chi = self.xi /(self.xi + np.sqrt(env.vpd))
        

        # These values need to retain any
        # dimensions of the original inputs - if ftemp_kphio is set to 1.0
        # (i.e. no temperature correction) then the dimensions of tc are lost
        # and the input to soilmstress might be scalar, so enforce the shape.
        # Note that rpmodel_1.0.6 collapses array inputs at this point.

    #    if self.shape == 1:
    #        self.mc = 1.0
    #        self.mj = 1.0
    #        self.mjoc = 1.0
    #    else:
    #        self.mc = np.ones(self.shape)
    #        self.mj = np.ones(self.shape)
    #        self.mjoc = np.ones(self.shape)
            
        # mj is equal to 1 as gammastar is null

        self.mj = 1.0

        # alternative variables
        kappa = env.kmm / env.ca

        # mc and mj:mv
        self.mc = (self.chi) / (self.chi + kappa)
        self.mjoc = (self.chi + kappa) / (self.chi)
        

    def c3(self, env, rootzonestress):
        r"""This method calculates key variables as follows:

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
                       {c_a + 2 \Gamma^{*} + 3 \Gamma^{*}
                       \sqrt{\frac{1.6 D \eta^{*}}{\beta(K + \Gamma^{*})}}}

        Finally,  :math:`m_c` is calculated, following Equation 7 in
        :cite:`Stocker:2020dh`, as:

        .. math::

            m_c = \frac{c_i - \Gamma^{*}}{c_i + K},

        where :math:`K` is the Michaelis Menten coefficient of Rubisco-limited
        assimilation.
        """
        
        ## Modified version
        ## Calculation beta as a function of theta following Lavergne et al. 2020 (Fig 6a, GCB)
        beta_c3 = np.exp(1.73 * env.theta + 4.55)
        #beta_c3 = 146
        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        self.xi = np.sqrt((beta_c3 *
                           (env.kmm + env.gammastar))
                          / (1.6 * env.ns_star))

        self.chi = (env.gammastar / env.ca +
                    (1.0 - env.gammastar / env.ca) * self.xi /
                    (self.xi + np.sqrt(env.vpd)))
         
         
        ## Original version
        #self.xi = np.sqrt((self.pmodel_params.beta_cost_ratio_c3 * rootzonestress *
        #                   (env.kmm + env.gammastar))
        #                  / (1.6 * env.ns_star))

        #self.chi = (env.gammastar / env.ca +
        #            (1.0 - env.gammastar / env.ca) * self.xi /
        #            (self.xi + np.sqrt(env.vpd)))

        # Define variable substitutes:
        vdcg = env.ca - env.gammastar
        vacg = env.ca + 2.0 * env.gammastar
        vbkg = (beta_c3 *
                (env.kmm + env.gammastar))
        #vbkg = (self.pmodel_params.beta_cost_ratio_c3 *
        #        rootzonestress *
        #        (env.kmm + env.gammastar))
                
        # Calculate mj
        # NOTE: this differs from rpmodel, which uses length not dim here, so
        # unwrapped matrix inputs. Also, rpmodel includes a check for vpd > 0,
        # but this is guaranteed by clip above (also true in rpmodel).

        vsr = np.sqrt(1.6 * env.ns_star * env.vpd / vbkg)
        mj = vdcg / (vacg + 3.0 * env.gammastar * vsr)

        # Mask values with ns star <= 0 and vbkg <=0 - need an array for this
        mask = np.logical_and(env.ns_star <= 0, vbkg <= 0)
        mj = np.array(mj)
        mj[mask] = np.nan 
        # np.where _always_ returns an array, so catch scalars.
        self.mj = mj.item() if np.ndim(mj) == 0 else mj

        # alternative variables
        gamma = env.gammastar / env.ca
        kappa = env.kmm / env.ca

        # mc and mj:mv
        self.mc = (self.chi - gamma) / (self.chi + kappa)
        self.mjoc = (self.chi + kappa) / (self.chi + 2 * gamma)

    def summarize(self, dp=2):
        """Prints a summary of the variables calculated within an instance
        of CalcOptimalChi including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.

        Returns:
            None
        """

        attrs = [('xi', ' Pa ^ (1/2)'), 
                 ('chi', '-'), 
                 ('mc', '-'),
                 ('mj', '-'),
                 ('mjoc', '-')]
        summarize_attrs(self, attrs, dp=dp)


class CalcLUEVcmax:
    r"""Estimate light use efficiency and maximum carboxylation rate
    :math:`V_{cmax}`. The class implements:

    - :math:`J_{max}` limitation of light use efficiency, providing two
      approaches (``wang17`` and ``smith19``),
    - soil moisture stress limitation, and
    - temperature dependence of apparent quantum yield efficiency.

    Light use efficiency (LUE) is calculated from the inputs as:

    .. math::

        \text{LUE} = \phi_0 \cdot \phi_0(T) \cdot  m_j \cdot m_{jlim} \cdot M_C \cdot \beta

    The Rubisco carboxylation capacity (:math:`V_{cmax}`) of the system is then back
    calculated from LUE as:

    .. math::

          V_{cmax} = \frac{\text{LUE}}{m_c M_C}

    Attributes:

        optchi (:class:`CalcOptimalChi`): an instance of :class:`CalcOptimalChi`
            providing the :math:`\ce{CO2}` limitation term of light use efficiency
            (:math:`\m_j`) and the  the :math:`\ce{CO2}` limitation term for
            Rubisco assimilation (:math:`m_c`).
        kphio (float): The apparent quantum yield efficiency (:math:`\phi_0`,
            unitless).
        ftemp_kphio (float): A factor to capture the temperature dependence of
            quantum yield efficiency (:math:`\phi_0(T)`), defaulting to 1.0 for
            no temperature dependence (see :func:`calc_ftemp_kphio`).
        soilmstress (float): A factor to capture the soil moisture stress
            (:math:`\beta`), defaulting to 1.0 for no soil moisture stress
            (see :func:`calc_soilmstress`).
        method (str): method to apply :math:`J_{max}` limitation (default: ``wang17``,
            or ``smith19`` or ``none``)
        mjlim (float): :math:`J_{max}` limitation factor, calculated using the method.
        lue (float): calculated light use efficiency per unit absolute irradiance.
        vcmax (float): calculated maximum carboxylation rate per unit absolute
            irradiance.
        omega (float): component of :math:`J_{max}` calculation (:cite:`Smith:2019dv`).
        omega_star (float):  component of :math:`J_{max}` calculation (:cite:`Smith:2019dv`).

    Other Parameters:

        c_molmass: the molar mass of carbon (:math:`M_C`, `pmodel_params.k_c_molmass`)



    Examples:

        >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000, theta=0.4, d13CO2= -8.4, D14CO2 = 19.2)
        >>> optchi = CalcOptimalChi(env)
        >>> # Using Wang et al 2017
        >>> out_wang = CalcLUEVcmax(optchi, kphio = 0.081785, ftemp_kphio = 0.656,
        ...                         soilmstress = 1, method='wang17')
        >>> round(out_wang.lue, 5)
        0.25998
        >>> round(out_wang.vcmax, 6)
        0.06305
        >>> # Using Smith et al 2019
        >>> out_smith = CalcLUEVcmax(optchi, kphio = 0.081785, ftemp_kphio = 0.656,
        ...                          soilmstress = 1, method='smith19')
        >>> round(out_smith.lue, 6)
        0.087808
        >>> round(out_smith.vcmax, 6)
        0.021295
        >>> round(out_smith.omega, 5)
        1.10847
        >>> round(out_smith.omega_star, 5)
        1.28576
        >>> # No Jmax limitation
        >>> out_none = CalcLUEVcmax(optchi, kphio = 0.081785, ftemp_kphio = 0.656,
        ...                    soilmstress = 1, method='none')
        >>> round(out_none.lue, 6)
        0.464392
        >>> round(out_none.vcmax, 6)
        0.112623

    """

    # TODO - apparent incorrectness of wang and smith methods with _ca_ variation,
    #        work well with varying temperature but not _ca_ variation (or
    #        e.g. elevation gradient David Sandoval, REALM meeting, Dec 2020)

    def __init__(self, optchi: CalcOptimalChi,
                 kphio: Union[float, np.ndarray],
                 ftemp_kphio: Union[float, np.ndarray] = 1.0,
                 soilmstress: Union[float, np.ndarray] = 1.0,
                 method: str = 'wang17',
                 pmodel_params: PModelParams = PModelParams()
                 ):

        self.shape = check_input_shapes(optchi.mj, optchi.mjoc, kphio,
                                        ftemp_kphio, soilmstress)

        self.optchi = optchi
        self.kphio = kphio
        self.ftemp_kphio = ftemp_kphio
        self.soilmstress = soilmstress
        self.method = method
        self.pmodel_params = pmodel_params
        self.mjlim = None
        self.lue = None
        self.vcmax = None
        self.omega = None
        self.omega_star = None

        all_methods = {'wang17': self.wang17,
                       'smith19': self.smith19,
                       'none': self.none}

        if self.method == 'c4':
            raise ValueError('This class does not implement a fixed method for C4 '
                             'photosynthesis. To replicate rpmodel choose c4=True and '
                             'method="none"')

        if self.method in all_methods:

            # Use the selected method to calculate limitation factors
            this_method = all_methods[self.method]
            this_method()

            # Now calculate the LUE and V_cmax
            # Light use efficiency (gpp per unit absorbed light)
            self.lue = (self.kphio * self.ftemp_kphio * self.optchi.mj * self.mjlim *
                        self.pmodel_params.k_c_molmass * self.soilmstress)

            # Back calculate Vcmax normalised per unit absorbed PPFD (assuming iabs=1)
            self.vcmax = self.lue / (self.optchi.mc * self.pmodel_params.k_c_molmass)

        else:
            raise ValueError(f"CalcLUEVcmax: method argument '{method}' invalid.")

    def __repr__(self):

        return (f"CalcLUEVCmax(lue={self.lue}, vcmax={self.vcmax}, "
                f"mjlim={self.mjlim}, omega={self.omega}, omega_star={self.omega_star})")

    def wang17(self):
        r"""Calculate a :math:`J_{max}` limitation following
        :cite:`Wang:2017go`. The factor is described in Equation 49 of
        :cite:`Wang:2017go` and is the square root term at the end of that
        equation:

        .. math::

            m_{jmax} = \sqrt{1- \left(\frac{c^*}{m_j}\right)^{\frac{2}{3}}}

        Other parameters:

            cstar: A cost parameter for maintaining :math:`J_{max}`
                (:math:`c^*`, `pmodel_params.wang_c`)

        """

        # Calculate mjlim (square root term in Eqn 2 of Wang et al 2017)
        vals = 1 - (self.pmodel_params.wang17_c / self.optchi.mj) ** (2.0 / 3.0)

        # Convert to array if needed and handle negative and nan values
        vals = np.array(vals) if np.ndim(vals) == 0 else vals
        mask = vals >= 0  # Also traps np.nan
        mjlim = np.empty_like(vals)
        mjlim[mask] = np.sqrt(vals[mask])
        mjlim[~ mask] = np.nan

        # revert scalars back to a scalar value
        self.mjlim = mjlim.item() if np.ndim(mjlim) == 0 else mjlim

    def smith19(self):
        r"""Calculate a :math:`J_{max}` limitation following
        :cite:`Smith:2019dv`. The value of :math:`m_{jlim}` is taken as the
        final term of  Equation 18 of :cite:`Smith:2019dv`:

        .. math::
            :nowrap:

            \[
                \begin{align*}
                m_{jlim} &= \frac{\omega^*}{8\theta}, \text{where} \\
                \omega^* &= 1 + \omega - \sqrt{(1 + \omega) ^2 -4\theta\omega}, \text{and}\\
                \omega &= (1 - 2\theta) + \sqrt{(1-\theta)
                    \left(\frac{1}{\frac{4c}{m}(1 - \theta\frac{4c}{m})}-4\theta\right)}
                \end{align*}
            \]

        Other parameters:

            theta: A term to capture the curved relationship between light intensity
                and photosynthetic capacity :math:`J_{max}` (:math:`\theta`, `pmodel_params.smith19_theta`)
            c_cost: A cost parameter for maintaining :math:`J_{max}`
                (:math:`c`, `pmodel_params.smith19_c_cost`)

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
        omega = np.where(self.optchi.mj < m_star,
                         -(1 - (2 * theta)) - np.sqrt((1 - theta) * v),
                         -(1 - (2 * theta)) + np.sqrt((1 - theta) * v))

        # np.where _always_ returns an array, so catch scalars
        self.omega = omega.item() if np.ndim(omega) == 0 else omega

        self.omega_star = (1.0 + self.omega -  # Eq. 18
                           np.sqrt((1.0 + self.omega) ** 2 -
                                   (4.0 * theta * self.omega)))

        # Effect of Jmax limitation
        self.mjlim = self.omega_star / (8.0 * theta)

    def none(self):
        """No :math:`J_{max}` limitation (:math:`m_{jlim} = 1.0`)
        """

        # Set Jmax limitation to unity - could define as 1.0 in __init__ and
        # pass here, but setting explicitly within the method for clarity.
        self.mjlim = 1.0


class CalcCarbonIsotopes:
    r"""Calculate the discrimination against 13C and 14C for C3 and C4 plants and associated carbon isotopic composition (d13C) in permil

    The chosen method is automatically used to estimate these values when an
    instance is created.

    Attributes:

        gammastar (float): the photorespiratory :math:`\ce{CO2}` compensation point
            (:math:`\Gamma^{*}`, see :func:`calc_gammastar`).
        ca (float): the ambient partial pressure of :math:`\ce{CO2}` (:math:`c_a`,
            see :func:`calc_co2_to_ca`)
        method (str): one of ``c4`` or ``c3``
        chi (float): the ratio of leaf internal to ambient :math:`\ce{CO2}`
            partial pressure (:math:`\chi`).


    Returns:

        An instance of :class:`CalcCarbonIsotopes` where the :attr:`Delta13C_simple`, :attr:`Delta13C`,
        :attr:`Delta14C`,:attr:`d13C_leaf`,:attr:`d14C_leaf`,:attr:`d13C_wood`
        have been populated using the chosen method.
          
    Examples:

        >>> env = PModelEnvironment(tc= 20, patm=101325, co2=400, vpd=1000, theta=0.4, d13CO2= -8.4, D14CO2 = 19.2)
        >>> mod_c3 = PModel(env,c4=False)
        >>> mod_c4 = PModel(env,c4=True)
        >>> Ciso_c3 = mod_c3.delta
        >>> round(Ciso_c3.Delta13C, 2)
        20.41
        >>> round(Ciso_c3.d13C_leaf, 2)
        -28.23
        >>> Ciso_c4 = mod_c4.delta
        >>> round(Ciso_c4.Delta13C, 2)
        5.11
        >>> round(Ciso_c4.d13C_leaf, 2)
        -13.44
    """

    # TODO - move chi calc into __init__? Shared between the two methods

    def __init__(self, optchi: CalcOptimalChi,
                 gammastar: Union[float, np.ndarray],
                 ca: Union[float, np.ndarray],
                 tc: Union[float, np.ndarray],
                 d13CO2: Union[float, np.ndarray],
                 D14CO2: Union[float, np.ndarray],
                 method: str = 'c3',
                 pmodel_params: PModelParams = PModelParams()
                 ):

        # Check inputs are broadcastable
        self.shape = check_input_shapes(gammastar, ca, tc, d13CO2, D14CO2, optchi.chi)

        self.optchi = optchi
        self.gammastar = gammastar
        self.ca = ca
        self.tc = tc
        self.d13CO2 = d13CO2
        self.D14CO2 = D14CO2
        
        # Identify and run the selected method
        self.pmodel_params = pmodel_params
        self.method = method
        all_methods = {'c3': self.c3, 'c4': self.c4}

        if self.method in all_methods:
            this_method = all_methods[self.method]
            this_method(gammastar=gammastar, ca=ca, tc=tc, d13CO2=d13CO2, D14CO2=D14CO2, chi=optchi)
        else:
            raise ValueError(f"CalcCarbonIsotopes: method argument '{method}' invalid.")


    def __repr__(self):

        return f"CalcCarbonIsotopes(delta={self.Delta13C})"

    def c4(self, **kwargs):
        r""" :math:`\delta 13C` is calculated following simple model,
        but using a C4 specific estimate of the unit cost ratio :math:`\beta`,
        specified in :meth:`~pyrealm.param_classes.PModelParams.beta_cost_ratio_c4`.

        """

        # 13C discrimination (‰): von Caemmerer et al. (2014) Eq. 1
        self.Delta13C_simple = self.pmodel_params.farquhar_a + (self.pmodel_params.vonCaemmerer_b4 + (self.pmodel_params.farquhar_b - self.pmodel_params.vonCaemmerer_s)*self.pmodel_params.vonCaemmerer_phi - self.pmodel_params.farquhar_a)*self.optchi.chi
        
        ## Equation A5 from von Caemmerer et al. (2014)
        #b4 = (-9.483*1000)/ (273 + self.tc) + 23.89 + 2.2
        b4 = self.pmodel_params.vonCaemmerer_b4
        
        self.Delta13C = self.pmodel_params.farquhar_a + (b4 + (self.pmodel_params.farquhar_b - self.pmodel_params.vonCaemmerer_s)*self.pmodel_params.vonCaemmerer_phi - self.pmodel_params.farquhar_a)*self.optchi.chi
        
        self.Delta14C = self.Delta13C*2
        
        ## Isotopic composition of leaf
        self.d13C_leaf = (self.d13CO2 - self.Delta13C)/(1 + self.Delta13C/1000)
        
        self.d14C_leaf = (self.D14CO2 - self.Delta14C)/(1 + self.Delta14C/1000)
        
        ## Isotopic composition of wood considering post-photosynthetic fractionation: estimated around 2.1‰ (Frank et al. 2015 NCC)
        self.d13C_wood = self.d13C_leaf + self.pmodel_params.frank_postfrac
                
                
    def c3(self, **kwargs):
        r"""This method calculates key variables as follows:

        :math:`\Delta 13C simple` is calculated following simple model.
        :math:`\Delta 13C` is calculated following model with photorespiratory effect.
        :math:`\Delta 14C` is calculated as twice the `\Delta 13C` .
        :math:`\d13C` is calculated from `\Delta 13C` and `\d13CO2` (input).
        :math:`\d14C` is calculated from `\Delta 14C` and `\D14CO2` (input).
        """

        # 13C discrimination (permil): Farquhar et al. (1982)
        ## Simple
        self.Delta13C_simple = self.pmodel_params.farquhar_a + (self.pmodel_params.farquhar_b - self.pmodel_params.farquhar_a)*self.optchi.chi
        
        ## with photorespiratory effect:
        self.Delta13C = self.pmodel_params.farquhar_a + (self.pmodel_params.farquhar_b2 - self.pmodel_params.farquhar_a)*self.optchi.chi - self.pmodel_params.farquhar_f*self.gammastar / self.ca
        
        ## 14C discrimination is twice the 13C discrimination (Graven et al. 2020)
        self.Delta14C = self.Delta13C*2
        
        
        ## Isotopic composition of leaf
        self.d13C_leaf = (self.d13CO2 - self.Delta13C)/(1 + self.Delta13C/1000)
                
        self.d14C_leaf = (self.D14CO2 - self.Delta14C)/(1 + self.Delta14C/1000)
        
        ## Isotopic composition of wood considering post-photosynthetic fractionation: 
        self.d13C_wood = self.d13C_leaf + self.pmodel_params.frank_postfrac
        
                
    def summarize(self, dp=2):
        """Prints a summary of the variables calculated within an instance
        of CalcCarbonIsotopes including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.

        Returns:
            None
        """

        attrs = ['Delta13C_simple','Delta13C','Delta14C','d13C_leaf','d14C_leaf','d13C_wood']
        summarize_attrs(self, attrs, dp=dp)
