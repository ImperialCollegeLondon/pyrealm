import numpy
from typing import Optional
import warnings
import dotmap
from pypmodel.params import PARAM


def calc_density_h2o(tc: float, p: float) -> float:
    """*Density of water*

    Calculates the density of water as a function of temperature and atmospheric
    pressure, using the Tumlirz Equation.

    Example:
        >>> calc_density_h2o(20, 101325) # doctest: +ELLIPSIS
        998.2056...

    References:
        F.H. Fisher and O.E Dial, Jr. (1975) Equation of state of pure water and
        sea water, Tech. Rept., Marine Physical Laboratory, San Diego, CA.

    Parameters:
        tc: air temperature, °C
        p: atmospheric pressure, Pa

    Returns:
        Water density as a float in (g cm^-3)
    """

    # Get powers of tc, including tc^0 = 1 for constant terms
    tc_pow = tc ** numpy.arange(0, 10)

    # Calculate lambda, (bar cm^3)/g:
    lambda_val = numpy.sum(numpy.array(PARAM.FisherDial.lambda_) * tc_pow[:5])

    # Calculate po, bar
    po_val = numpy.sum(numpy.array(PARAM.FisherDial.po) * tc_pow[:5])

    # Calculate vinf, cm^3/g
    vinf_val = numpy.sum(numpy.array(PARAM.FisherDial.vinf) * tc_pow)

    # Convert pressure to bars (1 bar <- 100000 Pa)
    pbar = 1e-5 * p

    # Calculate the specific volume (cm^3 g^-1):
    spec_vol = vinf_val + lambda_val / (po_val + pbar)

    # Convert to density (g cm^-3) -> 1000 g/kg; 1000000 cm^3/m^3 -> kg/m^3:
    rho = 1e3 / spec_vol

    return rho


def calc_ftemp_arrh(tk: float, dha: float) -> float:
    r"""**Temperature scaling of enzyme kinetics**

    Calculates the temperature-scaling factor :math:`f` for enzyme kinetics
    following an Arrhenius response for a given temperature (``tk``, :math:`T`)
    and activation energy (``dha``, :math:`\Delta H_a`) relative to the standard
    reference temperature :math:`T_0` and given the universal gas constant
    :math:`R`.

    Arrhenius kinetics are described as:

    .. math::

        x(T)= exp(c - \Delta H_a / (T R))

    The temperature-correction function :math:`f(T, \Delta H_a)` is therefore:

    .. math::
        :nowrap:

            \begin{align*}
                f &= \frac{x(T)}{x(T_0)} \\
                  &= exp \left( \frac{\Delta H_a (T - T_0)}{T_0 R T}\right)\text{, or equivalently}\\
                  &= exp \left( \frac{\Delta H_a}{R} \cdot \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
            \end{align*}

    Examples:
        >>> # Relative rate change from 25 to 10 degrees Celsius (percent change)
        >>> (1.0-calc_ftemp_arrh( 283.15, 100000)) * 100 # doctest: +ELLIPSIS
        88.19912...

    Args:
        tk: Temperature (Kelvin)
        dha: Activation energy (J mol-1)

    Returns:
        A float value for :math:`f`
    """

    # Note that the following forms are equivalent:
    # exp( dha * (tk - 298.15) / (298.15 * kR * tk) )
    # exp( dha * (tc - 25.0)/(298.15 * kR * (tc + 273.15)) )
    # exp( (dha/kR) * (1/298.15 - 1/tk) )

    tkref = PARAM.k.To + PARAM.k.CtoK

    return numpy.exp(dha * (tk - tkref) / (tkref * PARAM.k.R * tk))


def calc_ftemp_inst_rd(tc: float) -> float:
    """*Temperature response of dark respiration*

    This function calculates the temperature-scaling factor for dark respiration
    at a given temperature :math:`T` in °C, relative to the standard reference
    temperature :math:`T0`, following Heskel et al. 2016 (Table1, Eqn 2). The
    default parameterisation uses the global means(:math:`b = 0.1012, c = 0.0005`)

    .. math::

            fr = exp( b (T0 - T) -  c ( T0^2 - T^2 ))

    The default parameters are

    References:
         Heskel,  M.,  O’Sullivan,  O.,  Reich,  P.,  Tjoelker,  M.,
         Weerasinghe,  L.,  Penillard,  A.,Egerton, J., Creek, D.,
         Bloomfield, K., Xiang, J., Sinca, F., Stangl, Z.,
         Martinez-De La Torre, A., Griffin, K., Huntingford, C., Hurry, V.,
         Meir, P., Turnbull, M.,and Atkin, O. (2016)  Convergence in the
         temperature response of leaf respiration across biomes and plant
         functional types, Proceedings of the National Academy of Sciences,
         113,  3832–3837,  doi:10.1073/pnas.1520282113

    >>> # Relative percentage instantaneous change in Rd going from 10 to 25 degrees
    >>> (calc_ftemp_inst_rd(25) / calc_ftemp_inst_rd(10) - 1) * 100 # doctest: +ELLIPSIS
    250.95927...

    Args:
        tc: Temperature (degrees Celsius)

    Returns:
        A float value for \eqn{fr}
    """

    return numpy.exp(PARAM.Heskel.b * (tc - PARAM.k.To) -
                     PARAM.Heskel.c * (tc ** 2 - PARAM.k.To ** 2))


def calc_ftemp_inst_vcmax(tcleaf: float, tcgrowth: float = None) -> float:
    """*Instantaneous temperature response of* :math:`V_{cmax}`

    This function calculates the temperature-scaling factor of the instantaneous
    temperature response of :math:`V_{cmax}` relative to the standard reference
    temperature, following modified Arrhenius kinetics based on Kattge & Knorr
    (2007). Calculates \eqn{f} for the conversion:

    .. math::

       V = f V_{ref}

    Details:
        The function is given by Kattge & Knorr (2007) as

        .. math::

            fv = f(T, \Delta Hv) A/B

        where :math:`f(T, \Delta Hv)` is a regular Arrhenius-type temperature
        response function (see `calc_ftemp_arrh`) with :math:`Hv=71513` J mol-1,

        .. math::

            A = 1 + exp( (T0 \Delta S - Hd) / (T0 R) )

        and

        .. math::

            B = 1 + exp( (T \Delta S - Hd) / (TK R) )

        Here, :math:`T` is in Kelvin, :math:`T0=293.15` K, :math:`Hd = 200000` J mol-1
        is the deactivation energy and :math:`R` is the universal gas constant and
        is 8.3145 J mol-1 K-1, and

        .. math::

            \Delta S = aS - bS T

        with :math:`aS = 668.39` J mol-1 K-1, and :math:`bS = 1.07` J mol-1 K-2,
        and :math:`T` given in degrees Celsius (!)

    References:
        Kattge, J. and Knorr, W.:  Temperature acclimation in a biochemical model
        of photosynthesis: a reanalysis of data from 36 species, Plant, Cell
        and Environment, 30,1176–1190, 2007.

    >>> # Relative change in Vcmax going (instantaneously, i.e. not
    >>> # not acclimatedly) from 10 to 25 degrees (percent change):
    >>> ((calc_ftemp_inst_vcmax(25)/calc_ftemp_inst_vcmax(10)-1) * 100 ) # doctest: +ELLIPSIS
    283.17753...

    Args:
        tcleaf: Leaf temperature, or in general the temperature relevant for
            photosynthesis (°C)
        tcgrowth: Growth temperature (°C). In the P-model, taken to be equal
            to `tcleaf`, as in the defaults here.

    Returns:

    """

    if tcgrowth is None:
        tcgrowth = tcleaf

    # Convert temperatures to Kelvin
    tkref = PARAM.k.To + 273.15
    tkleaf = tcleaf + 273.15

    # Calculate entropy following Kattge & Knorr (2007): slope and intercept
    # are defined using temperature in °C, not K!!! 'tcgrowth' corresponds
    # to 'tmean' in Nicks, 'tc25' is 'to' in Nick's
    dent = PARAM.KattgeKnorr.a_ent + PARAM.KattgeKnorr.b_ent * tcgrowth
    fva = calc_ftemp_arrh(tkleaf, PARAM.Rubisco.Ha)
    fvb = ((1 + numpy.exp((tkref * dent - PARAM.Rubisco.Hd) /
                          (PARAM.k.R * tkref))) /
           (1 + numpy.exp((tkleaf * dent - PARAM.Rubisco.Hd) /
                          (PARAM.k.R * tkleaf))))

    return fva * fvb


def calc_ftemp_kphio(tc: float, c4: bool = False) -> float:
    """*Temperature dependence of the quantum yield efficiency*

    Calculates the temperature dependence of the quantum yield efficiency,
    using a quadratic function. The default parameterisations are below.

    C3 plants:
        The temperature dependence of the maximum quantum yield of photosystem
        II in light-adapted tobacco leaves, determined by Bernacchi et al.
        (2003).

        .. math::

            \phi(T) = 0.352 + 0.022 T - 0.00034 T^2

    C4 plants:
         C4 photosynthesis is calculated based on unpublished work as:

         .. math::
            \phi(T) = -0.008 + 0.00375 T - 0.000058 T^2

    The factor :math:`\phi(T)` is to be multiplied with leaf absorptance and
    the fraction of absorbed light that reaches photosystem II. In the
    P-model these additional factors are lumped into a single apparent
    quantum yield efficiency parameter (argument `kphio` to function
    :func:`rpmodel`).

    References:
        Bernacchi, C. J., Pimentel, C., and Long, S. P.:  In vivo temperature
        response functions  of  parameters required  to  model  RuBP-limited
        photosynthesis,  Plant  Cell Environ., 26, 1419–1430, 2003

    >>> # Relative change in the quantum yield efficiency between 5 and 25
    >>> # degrees celsius (percent change):
    >>> (calc_ftemp_kphio(25.0) / calc_ftemp_kphio(5.0) - 1) * 100 # doctest: +ELLIPSIS
    52.03969...
    >>> # Relative change in the quantum yield efficiency between 5 and 25
    >>> # degrees celsius (percent change) for a C4 plant:
    >>> (calc_ftemp_kphio(25.0, c4=True) /
    ...  calc_ftemp_kphio(5.0, c4=True) - 1) * 100 # doctest: +ELLIPSIS
    432.25806...


    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        c4: Boolean specifying whether fitted temperature response for C4 plants
            is used. Defaults to \code{FALSE}.

    Returns:

    """

    if c4:
        coef = PARAM.kphio.C4
    else:
        coef = PARAM.kphio.C3

    return coef[0] + coef[1] * tc + coef[2] * tc ** 2


def calc_gammastar(tc: float, patm: float) -> float:
    r"""CO2 compensation point

    Calculates the photorespiratory CO2 compensation point in absence of dark
    respiration (:math:`\Gamma^{*}`, Farquhar, 1980) as:

    .. math::

        \Gamma^{*} = \Gamma^{*}_{0} \cdot \frac{p}{p_0} \cdot f(T, \Delta H_a)

    where:

    - :math:`T_0` and :math:`T` are the standard temperature and the local
      temperature set in ``tc``.
    - :math:`p_0` and :math:`p` are the standard pressure and the local
      pressure set in ``patm``.
    - :math:`\Gamma^{*}_{0}` is the reference value at standard temperature
      (:math:`T_0`) and pressure (:math:`P_0`), calculated by Bernacchi et al.
      (2001) as :math:`42.75 \mu\text{mol}\ \text{mol}^{-1}` and here converted
      using :math:`42.75 \cdot p_0 = 4.332 \text{Pa}`.
    - :math:`\Delta H_a` is the activation energy of the system and
      :math:`f(T, \Delta H_a)` modifies this to the the local temperature 
      following an Arrhenius-type temperature response function implemented
      in :func:`calc_ftemp_arrh`.

    References: 
        Farquhar,  G.  D.,  von  Caemmerer,  S.,  and  Berry,  J.  A.: A
        biochemical  model  of photosynthetic CO2 assimilation in leaves of C3
        species, Planta, 149, 78–90, 1980.

        Bernacchi,  C.  J.,  Singsaas,  E.  L.,  Pimentel,  C.,  Portis,  A.
        R.  J.,  and  Long,  S.  P.:Improved temperature response functions
        for models of Rubisco-limited photosyn-thesis, Plant, Cell and
        Environment, 24, 253–259, 2001

    Examples: 
        >>> # CO2 compensation point at 20 degrees Celsius and standard 
        >>> # atmosphere (in Pa) 
        >>> calc_gammastar(20, 101325) # doctest: +ELLIPSIS
        3.33925...

    Args: 
        tc: Temperature, relevant for photosynthesis (°C) 
        patm: Atmospheric pressure (Pa)

    Returns: 
        A float value for :math:`\Gamma^{*}` (in Pa)
    """

    return (PARAM.Bernacci.gs25_0 * patm / PARAM.k.Po *
            calc_ftemp_arrh((tc + PARAM.k.CtoK), dha=PARAM.Bernacci.dha))


def calc_kmm(tc: float, patm: float) -> float:
    r"""**Michaelis Menten coefficient for Rubisco-limited photosynthesis**
    
    Calculates the Michaelis Menten coefficient of Rubisco-limited assimilation
    as a function of temperature (``tc``, :math:`T`) and atmospheric pressure
    (``patm``, :math:`p`).

    - The partial pressure of oxygen at :math:`p` is 
      :math:`p_{O_{2}} = 0.209476 \cdot p`.
    - The function :math:`f(T, \Delta H)` is an Arrhenius-type temperature 
      response function, implemented as :func:`pypmodel.pmodel.calc_ftemp_arrh`,
      and used here to correct activation energies at standard temperature for
      both carbon and oxygen to the local temperature:

      .. math::
        :nowrap:

        \[ 
            \begin{align*}
                K_c &= K_{c25} \cdot f(T, \Delta H_{kc})\\
                K_o &= K_{o25} \cdot f(T, \Delta H_{ko})
            \end{align*}
        \]

    - And hence, the Michaelis-Menten coefficient :math:`K` of Rubisco-limited
      photosynthesis (Farquhar, 1980) is calculated as:

      .. math::
          K = K_c ( 1 + p_{O_{2}} / K_o),

    The default values set in the parameterisation are taken from Bernacchi
    et al. (2001). :math:`K_{c25}` and :math:`K_{c25}` are converted from 
    molar values using atmospheric pressure at a height of 227.076 metres:
    ``calc_patm(227.076) = 98716.403 Pa``.

    .. TODO - why this height?

    - :math:`\Delta H_{kc} = 79430 J mol-1`,
    - :math:`\Delta H_{ko} = 36380 J mol-1`,
    - :math:`K_{c25} = 404.9 \mu\text{mol}\ \text{mol}^{-1}` or 
      :math:`404.9 \cdot 98716.403 = 39.97 \text{Pa}`, and
    - :math:`K_{o25} = 278.4 \text{mmol}\ \text{mol}^{-1}` or 
      :math:`278.4 \cdot 98716.403 = 27480 \text{Pa}`.
 
    References:

        Farquhar,  G.  D.,  von  Caemmerer,  S.,  and  Berry,  J.  A. (1980) A
        biochemical  model  of photosynthetic CO2 assimilation in leaves of C3
        species, Planta, 149, 78–90
    
        Bernacchi,  C.  J.,  Singsaas,  E.  L.,  Pimentel,  C.,  Portis,  A.
        R.  J.,  and  Long,  S.  P. (2001) Improved temperature response functions
        for models of Rubisco-limited photosynthesis, Plant, Cell and
        Environment, 24, 253–259
    
    Examples:

        >>> # Michaelis-Menten coefficient at 20 degrees Celsius and standard 
        >>> # atmosphere (in Pa):
        >>> calc_kmm(20, 101325) # doctest: +ELLIPSIS
        46.09927...
    
    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        patm: Atmospheric pressure (Pa)

    Returns:
        A numeric value for \eqn{K} (in Pa)
    """
    
    # conversion to Kelvin
    tk = tc + PARAM.k.CtoK

    kc = PARAM.Bernacci.kc25 * calc_ftemp_arrh(tk, dha=PARAM.Bernacci.dhac)
    ko = PARAM.Bernacci.ko25 * calc_ftemp_arrh(tk, dha=PARAM.Bernacci.dhao)

    # O2 partial pressure
    po = PARAM.k.co * 1e-6 * patm

    return kc * (1.0 + po/ko)


def calc_soilmstress(soilm: float, meanalpha: float = 1.0):
    r"""*Empirical soil moisture stress factor*

    Calculates an empirical soil moisture stress factor  :math:`\beta` as a
    function of relative soil moisture (:math:`m_s`, fraction of field
    capacity) and average aridity, quantified by the local annual mean ratio
    of actual over potential evapotranspiration (:math:`\bar{\alpha}`).

    The value of :math:`\beta` is defined relative to two soil moisture
    thresholds: :math:`\theta_0 = 0, \theta^{*}=0.6` as:

    .. math::
        :nowrap:

        \[ \beta =
          \begin{cases}
            q(\theta_0 - \theta^{*})^2 + 1,  & \theta_0 < m_s <= \theta^{*} \\
            1, &  \theta^{*} < m_s,
          \end{cases}\]

    where :math:`q` is an aridity sensitivity parameter setting the stress
    factor at :math:`\theta_0`:

    .. math:: q=(1 - (a + b \bar{\alpha}))/(\theta^{*} - \theta_{0})^2,

    and :math:`a = 0.0` and :math:`b = 0.685` are empirically derived values.

    References:
         Stocker, B. et al. Geoscientific Model Development Discussions (in prep.)

    >>> # Relative reduction (%) in GPP due to soil moisture stress at
    >>> # relative soil water content ('soilm') of 0.2:
    >>> (calc_soilmstress(0.2) - 1) * 100 # doctest: +ELLIPSIS
    -13.99999...

    Args:
        soilm: Relative soil moisture as a fraction of field capacity
            (unitless). Defaults to 1.0 (no soil moisture stress).
        meanalpha: Local annual mean ratio of actual over potential
            evapotranspiration, measure for average aridity. Defaults to 1.0.

    Returns:
        A numeric value for \eqn{\beta}
    """

    if soilm > PARAM.soilmstress.thetastar:
        return 1.0
    else:
        y0 = (PARAM.soilmstress.a + PARAM.soilmstress.b * meanalpha)
        beta = (1.0 - y0) / (PARAM.soilmstress.theta0 - PARAM.soilmstress.thetastar) ** 2
        outstress = 1.0 - beta * (soilm - PARAM.soilmstress.thetastar) ** 2
        return max(0.0, min(1.0, outstress))


def calc_viscosity_h2o(tc: float, p: float):
    """Viscosity of water

    Calculates the viscosity of water as a function of temperature and atmospheric
    pressure.

    References:
        Huber, M. L., R. A. Perkins, A. Laesecke, D. G. Friend, J. V. Sengers,
        M. J. Assael, ..., K. Miyagawa (2009) New international formulation for
        the viscosity of H2O, J. Phys. Chem. Ref. Data, Vol. 38(2), pp. 101-125.

    >>> # Density of water at 20 degrees C and standard atmospheric pressure:
    >>> calc_viscosity_h2o(20, 101325) # doctest: +ELLIPSIS
    0.001001597...

    Args:
        tc: air temperature (°C)
        p: atmospheric pressure (Pa)

    Returns:
        A float giving the viscosity of water (mu, Pa s)
    """


    # Get the density of water, kg/m^3
    rho = calc_density_h2o(tc, p)

    # Calculate dimensionless parameters:
    tbar = (tc + PARAM.k.CtoK) / PARAM.Huber.tk_ast
    rbar = rho / PARAM.Huber.rho_ast

    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    tbar_pow = tbar ** numpy.arange(0, 4)
    mu0 = (1e2 * numpy.sqrt(tbar)) / numpy.sum(numpy.array(PARAM.Huber.H_i) / tbar_pow)

    # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
    h_array = numpy.array(PARAM.Huber.H_ij)
    ctbar = (1.0 / tbar) - 1.0
    row_j, _ = numpy.indices(h_array.shape) 
    mu1 = h_array * numpy.power(rbar - 1.0, row_j)
    mu1 = numpy.power(ctbar, numpy.arange(0, 6)) * numpy.sum(mu1, axis=0)
    mu1 = numpy.exp(rbar * mu1.sum())

    # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
    mu_bar = mu0 * mu1

    # Calculate mu (Eq. 1, Huber et al., 2009)
    return mu_bar * PARAM.Huber.mu_ast  # Pa s


def calc_patm(elv: float) -> float:
    """*Atmospheric pressure*

    Calculates atmospheric pressure as a function of elevation with reference to
    the standard atmosphere.  The elevation-dependence of atmospheric pressure
    is computed by assuming a linear decrease in temperature with elevation and
    a mean adiabatic lapse rate (Berberan-Santos et al., 1997, Eqn 3):

    .. math::

        p(z) = p_0 ( 1 - L z / T_0) ^{ g M / (R L) },

    where:

    - :math:`z` is the elevation above mean sea level (m, argument \code{elv}),
    - :math:`g` is the gravity constant (9.80665 m s-2),
    - :math:`p_0` is the standard atmospheric pressure at sea level,
    - :math:`L` is the mean adiabatic lapse rate (0.0065 K m-2),
    - :math:`M` is the molecular weight for dry air (0.028963 kg mol-1),
    - :math:`R` is the universal gas constant (8.3145 J mol-1 K-1), and
    - :math:`T_0` is the standard temperature in Kelvin (298.15 K, corresponds to 25 °C).

    References:
        Berberan-Santos, M. N., Bodunov, E. N., & Pogliani, L. (2009). On the
        barometric formula inside the Earth. Journal of Mathematical Chemistry,
        47(3), 990–1004. http://doi.org/10.1007/s10910-009-9620-7

    >>> # Standard atmospheric pressure, in Pa, corrected for 1000 m.a.s.l.
    >>> calc_patm(1000) # doctest: +ELLIPSIS
    90241.54...

    Args:
        elv: Elevation above sea-level (m.a.s.l.)

    Returns:
        A numeric value for \eqn{p}
    """

    # Convert elevation to pressure, Pa. This equation uses the base temperature
    # in Kelvins, while other functions use this constant in the PARAM units of
    # °C.

    kto = PARAM.k.To + PARAM.k.CtoK

    return (PARAM.k.Po * (1.0 - PARAM.k.L * elv / kto) **
            (PARAM.k.G * PARAM.k.Ma /
             (PARAM.k.R * PARAM.k.L)))


def pmodel(tc: float,
           vpd: float,
           co2: float,
           patm: Optional[float] = None,
           elv: Optional[float] = None,
           fapar: Optional[float] = None,
           ppfd: Optional[float] = None,
           soilm: float = None,
           meanalpha: float = None,
           kphio: Optional[float] = None,
           do_ftemp_kphio: bool = True,
           c4: bool = False,
           method_optci: str = "prentice14",
           method_jmaxlim: str = "wang17") -> dotmap.DotMap:

    r"""*Fit the  P-model*

    See the extended description in :ref:`pmodel` for detailed explanations
    of the parameter options and calculation of the P-model.

    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        vpd: Vapour pressure deficit (Pa)
        co2: Atmospheric CO2 concentration (ppm)
        patm: Atmospheric pressure (Pa).
        elv: Elevation above sea-level (m.a.s.l.).
        fapar: (Optional) Fraction of absorbed photosynthetically active radiation
            (unitless, defaults to None)
        ppfd: (Optional) Photosynthetic photon flux density
        kphio: (Optional) Apparent quantum yield efficiency (unitless).
        soilm: (Optional) Relative soil moisture as a fraction of field capacity (unitless).
        meanalpha: (Optional) Local annual mean ratio of actual over potential
            evapotranspiration, measure for average aridity.
        c4: (Optional) By default (`c4=False`), the C3 photosynthetic pathway is used.
        method_optci: (Optional) A character string specifying which method is to
            be used for calculating optimal ci:ca. Defaults to \code{"prentice14"}.
            Available also \code{"prentice14_num"} for a numerical solution to the
            same optimization criterium as  used for \code{"prentice14"}.
        method_jmaxlim: (Optional) Method for :math:`J_{max}` limitation,
            defaulting to `wang_17`.
        do_ftemp_kphio: (Optional) A logical specifying whether temperature-dependence
            of quantum yield efficiency after Bernacchi et al., 2003 is to be
            accounted for. Defaults to `TRUE`.


    Returns:
        A ``dotmap.DotMap`` object containing:

        - ``ca``: Ambient CO2 expressed as partial pressure (Pa)
        - ``gammastar``: Photorespiratory compensation point \eqn{\Gamma*}, (Pa, see :func:`calc_gammastar`).
        - ``kmm``: Michaelis-Menten coefficient :math:`K` for photosynthesis (Pa, see :func:`calc_kmm`).
        - ``ns_star``: Relative viscosity of water (unitless, see :ref:`ns_star`).
        - ``chi``: Optimal ratio of leaf internal to ambient CO2 (unitless, see :ref:`opt_chi`).
        - ``ci``: Leaf-internal CO2 partial pressure (Pa), calculated as \eqn{(\chi ca)}.

        - ``lue``: Light use efficiency (g C / mol photons), (see :ref:`lue`) 
        - ``mj``: Factor in the light-limited assimilation rate function, given by
                                \deqn{
                                    m = (ci - \Gamma*) / (ci + 2 \Gamma*)
                               }
                               where \eqn{\Gamma*} is given by \code{gammastar}.
        - ``mc``: Factor in the Rubisco-limited assimilation rate function, given by
                                \deqn{
                                    mc = (ci - \Gamma*) / (ci + K)
                               }
                               where \eqn{K} is given by \code{kmm}.
        - ``gpp``: Gross primary production (g C m-2), calculated as
                               \deqn{
                                   GPP = Iabs LUE
                               }
                               where \eqn{Iabs} is given by \code{fapar*ppfd} (arguments), and is
                               \code{NA} if \code{fapar==NA} or \code{ppfd==NA}. Note that \code{gpp} scales with
                               absorbed light. Thus, its units depend on the units in which \code{ppfd} is given.
        - ``iwue``: Intrinsic water use efficiency (iWUE, Pa), calculated as
                               \deqn{
                                     iWUE = ca (1-\chi)/(1.6)
                               }
        - ``gs``: Stomatal conductance (gs, in mol C m-2 Pa-1), calculated as
                               \deqn{
                                    gs = A / (ca (1-\chi))
                               }
                               where \eqn{A} is \code{gpp}\eqn{/Mc}.
        - ``vcmax``: Maximum carboxylation capacity \eqn{Vcmax} (mol C m-2) at growth temperature (argument
                              \code{tc}), calculated as
                              \deqn{
                                   Vcmax = \phi(T) \phi0 Iabs n
                              }
                              where \eqn{n} is given by \eqn{n=m'/mc}.
        - ``vcmax25``: Maximum carboxylation capacity \eqn{Vcmax} (mol C m-2) normalised to 25 deg C
                             following a modified Arrhenius equation, calculated as \eqn{Vcmax25 = Vcmax / fv},
                             where \eqn{fv} is the instantaneous temperature response by Vcmax and is implemented
                             by function \link{calc_ftemp_inst_vcmax}.
        - ``jmax``: The maximum rate of RuBP regeneration () at growth temperature (argument
                              \code{tc}), calculated using
                              \deqn{
                                   A_J = A_C
                              }
        - ``rd``: Dark respiration \eqn{Rd} (mol C m-2), calculated as
                             \deqn{
                                 Rd = b0 Vcmax (fr / fv)
                             }
                             where \eqn{b0} is a constant and set to 0.015 (Atkin et al., 2015), \eqn{fv} is the
                             instantaneous temperature response by Vcmax and is implemented by function
                             \link{calc_ftemp_inst_vcmax}, and \eqn{fr} is the instantaneous temperature response
                             of dark respiration following Heskel et al. (2016) and is implemented by function
                             \link{calc_ftemp_inst_rd}.

        The option ``method_jmaxlim=="smith19"`` adds the following:
        
        - ``omega``: Term corresponding to \eqn{\omega}, defined by Eq. 16 in Smith et al. (2019),
          and Eq. E19 in Stocker et al. (2019).
        - ``omega_star``: Term corresponding to \eqn{\omega^\ast}, defined by Eq. 18 in Smith et al.
          (2019), and Eq. E21 in Stocker et al. (2019).

    Examples:

        >>> mod_c3 = pmodel(tc=20, vpd=1000, co2=400, fapar=1, ppfd=300, elv=0)
        >>> # Key variables from pmodel
        >>> mod_c3.ci # doctest: +ELLIPSIS
        28.1420870...
        >>> mod_c3.ca # doctest: +ELLIPSIS
        40.52999...
        >>> mod_c3.chi # doctest: +ELLIPSIS
        0.694352...
        >>> mod_c3.gpp # doctest: +ELLIPSIS
        76.42544...
        >>> mod_c4 = pmodel(tc=20, vpd=1000, co2=400, fapar=1, ppfd=300, elv=0, c4=True)
        >>> # Key variables from pmodel
        >>> mod_c4.ci # doctest: +ELLIPSIS
        40.52999...
        >>> mod_c4.ca # doctest: +ELLIPSIS
        40.52999...
        >>> mod_c4.chi # doctest: +ELLIPSIS
        1.0...
        >>> mod_c4.gpp # doctest: +ELLIPSIS
        12.90735...

    References:
        Bernacchi, C. J., Pimentel, C., and Long, S. P.:  In vivo temperature response funtions  of  parameters
        required  to  model  RuBP-limited  photosynthesis,  Plant  Cell Environ., 26, 1419–1430, 2003

        Heskel,  M.,  O’Sullivan,  O.,  Reich,  P.,  Tjoelker,  M.,  Weerasinghe,  L.,  Penillard,  A.,
        Egerton, J., Creek, D., Bloomfield, K., Xiang, J., Sinca, F., Stangl, Z., Martinez-De La Torre, A.,
        Griffin, K., Huntingford, C., Hurry, V., Meir, P., Turnbull, M.,and Atkin, O.:  Convergence in the
        temperature response of leaf respiration across biomes and plant functional types, Proceedings of
        the National Academy of Sciences, 113,  3832–3837,  doi:10.1073/pnas.1520282113,2016.

        Huber,  M.  L.,  Perkins,  R.  A.,  Laesecke,  A.,  Friend,  D.  G.,  Sengers,  J.  V.,  Assael, M. J.,
        Metaxa, I. N., Vogel, E., Mares, R., and Miyagawa, K.:  New international formulation for the viscosity
        of H2O, Journal of Physical and Chemical ReferenceData, 38, 101–125, 2009

        Prentice,  I. C.,  Dong,  N.,  Gleason,  S. M.,  Maire,  V.,  and Wright,  I. J.:  Balancing the costs
        of carbon gain and water transport:  testing a new theoretical framework for  plant  functional  ecology,
        Ecology  Letters,  17,  82–91,  10.1111/ele.12211,http://dx.doi.org/10.1111/ele.12211, 2014.

        Wang, H., Prentice, I. C., Keenan, T. F., Davis, T. W., Wright, I. J., Cornwell, W. K.,Evans, B. J.,
        and Peng, C.:  Towards a universal model for carbon dioxide uptake by plants, Nat Plants, 3, 734–741, 2017.

        Atkin, O. K., et al.:  Global variability in leaf respiration in relation to climate, plant functional
        types and leaf traits, New Phytologist, 206, 614–636, doi:10.1111/nph.13253,

        Smith, N. G., Keenan, T. F., Colin Prentice, I. , Wang, H. , Wright, I. J., Niinemets, U. , Crous, K. Y.,
        Domingues, T. F., Guerrieri, R. , Yoko Ishida, F. , Kattge, J. , Kruger, E. L., Maire, V. , Rogers, A. ,
        Serbin, S. P., Tarvainen, L. , Togashi, H. F., Townsend, P. A., Wang, M. , Weerasinghe, L. K. and Zhou, S.
        (2019), Global photosynthetic capacity is optimized to the environment. Ecol Lett, 22: 506-517.
        doi:10.1111/ele.13210

        Stocker, B. et al. Geoscientific Model Development Discussions (in prep.)

    Returns:
        A named list of numeric values (including temperature and pressure dependent parameters of the
        photosynthesis model, P-model predictions, including all its corollary).
    """

    # Check arguments
    if patm is None and elv is None :
        raise ValueError('Provide either elevation (elv) or atmospheric pressure (patm)')

    if patm is None:
        warnings.warn("Atmospheric pressure (patm) not provided. Calculating it as a function of elevation "
                      "(elv), assuming standard atmosphere (101325 Pa at sea level).")
        patm = calc_patm(elv)

    # Soil moisture stress
    if soilm is None and meanalpha is None:
        soilm = 1.0
        meanalpha = 1.0
        do_soilmstress = False
    elif soilm is None or meanalpha is None:
        raise AttributeError('Provide both soilm and meanalpha to enable '
                             'soil moisture stress limitation.')
    else:
        do_soilmstress = True

    # kphio defaults:
    if kphio is None:
        if not do_ftemp_kphio:
            kphio = 0.049977
        elif do_soilmstress:
            kphio = 0.087182
        else:
            kphio = 0.081785

    # -----------------------------------------------------------------------
    # Temperature dependence of quantum yield efficiency
    # -----------------------------------------------------------------------
    # 'do_ftemp_kphio' is not actually a stress function, but is the temperature-
    # dependency of the quantum yield efficiency after Bernacchi et al., 2003

    if do_ftemp_kphio:
        ftemp_kphio = calc_ftemp_kphio(tc, c4)
    else:
        ftemp_kphio = 1.0

    # -----------------------------------------------------------------------
    # Calculate soil moisture stress as a function of soil moisture and mean alpha
    # -----------------------------------------------------------------------
    if do_soilmstress:
        soilmstress = calc_soilmstress(soilm, meanalpha)
    else:
        soilmstress = 1.0

    # -----------------------------------------------------------------------
    # Photosynthesis model parameters depending on temperature, pressure, and CO2.
    # -----------------------------------------------------------------------

    # ambient CO2 partial pressure (Pa)
    ca = co2_to_ca(co2, patm)
    # photorespiratory compensation point - Gamma-star (Pa)
    gammastar = calc_gammastar(tc, patm)
    # Michaelis-Menten coef. (Pa)
    kmm = calc_kmm(tc, patm)

    # viscosity correction factor relative to standards
    ns_star = calc_viscosity_h2o(tc, patm) / calc_viscosity_h2o(PARAM.k.To, PARAM.k.Po) # (unitless)

    # -----------------------------------------------------------------------
    # Optimal ci
    # The heart of the P-model: calculate ci:ca ratio (chi) and additional terms
    # -----------------------------------------------------------------------

    if c4:
        method_optci = "c4"

    out_optchi = CalcOptimalChi(kmm, gammastar, ns_star, ca, vpd, method=method_optci)

    # leaf-internal CO2 partial pressure (Pa)
    ci = out_optchi.chi * ca

    # -----------------------------------------------------------------------
    # Corrolary predictions
    # -----------------------------------------------------------------------
    # intrinsic water use efficiency (in Pa)
    iwue = (ca - ci) / 1.6

    # -----------------------------------------------------------------------
    # Vcmax and light use efficiency
    # -----------------------------------------------------------------------
    if c4:
        out_lue_vcmax = CalcLUEVcmax(out_optchi, kphio, ftemp_kphio,
                                     soilmstress, method='c4')
    else:
        out_lue_vcmax = CalcLUEVcmax(out_optchi, kphio, ftemp_kphio,
                                     soilmstress, method=method_jmaxlim)

    # -----------------------------------------------------------------------
    # Corrolary predictions
    # -----------------------------------------------------------------------

    # Vcmax25 (vcmax normalized to PARAM.k.To)
    ftemp25_inst_vcmax = calc_ftemp_inst_vcmax(tc, tc)
    vcmax25_unitiabs = out_lue_vcmax.vcmax_unitiabs / ftemp25_inst_vcmax

    # Dark respiration at growth temperature
    ftemp_inst_rd = calc_ftemp_inst_rd(tc)
    rd_unitiabs = (PARAM.Atkin.rd_to_vcmax * (ftemp_inst_rd / ftemp25_inst_vcmax) *
                   out_lue_vcmax.vcmax_unitiabs)

    # -----------------------------------------------------------------------
    # Quantities that scale linearly with absorbed light
    # -----------------------------------------------------------------------
    # Both fapar and ppfd are needed to calculate gpp, vcmax, rd and jmax, so
    # return None if those are not provided.
    if fapar is None or ppfd is None:
        gpp = None
        vcmax = None
        rd = None
        jmax = None
    else:
        # Scaling factor
        iabs = fapar * ppfd

        # Gross primary productivity
        gpp = iabs * out_lue_vcmax.lue  # in g C m-2 s-1

        # Vcmax per unit ground area is the product of the intrinsic quantum
        # efficiency, the absorbed PAR, and 'n'
        vcmax = iabs * out_lue_vcmax.vcmax_unitiabs

        # (vcmax normalized to 25 deg C)
        vcmax25 = iabs * vcmax25_unitiabs

        # Dark respiration
        rd = iabs * rd_unitiabs

        # Jmax using again A_J = A_C
        fact_jmaxlim = vcmax * (ci + 2.0 * gammastar) / (kphio * iabs * (ci + kmm))
        jmax = 4.0 * kphio * iabs / numpy.sqrt((1.0 / fact_jmaxlim) ** 2 - 1.0)

    # construct list for output
    out = dotmap.DotMap(ca=ca,
                        gammastar=gammastar,
                        kmm=kmm,
                        ns_star=ns_star,
                        chi=out_optchi.chi,
                        mj=out_optchi.mj,
                        mc=out_optchi.mc,
                        ci=ci,
                        lue=out_lue_vcmax.lue,
                        gpp=gpp,
                        iwue=iwue,
                        gs=numpy.infty if c4 else (gpp / PARAM.k.c_molmass) / (ca - ci),  # TODO - check with CP/BS
                        vcmax=vcmax,
                        vcmax25=vcmax25,
                        jmax=jmax,
                        rd=rd)

    return out


class CalcOptimalChi:
    """Returns an estimate of leaf internal to ambient CO2 partial pressure
    following the "simple formulation".

    TODO - fix this text

    Derived following Prentice et al.
                                 (2014) as:
                                 \deqn{
                                       \chi = \Gamma* / ca + (1- \Gamma* / ca) \\xi / (\\xi + \sqrt D )
                                 }
                                 with
                                 \deqn{
                                      \\xi = \sqrt (\beta (K+ \Gamma*) / (1.6 \eta*))
                                 }
                                 \eqn{\beta} is given by argument \code{beta}, \eqn{K} is \code{kmm} (see \link{calc_kmm}),
                                 \eqn{\Gamma*} is \code{gammastar} (see \link{calc_gammastar}). \eqn{\eta*} is \code{ns_star}.
                                 \eqn{D} is the vapour pressure deficit (argument \code{vpd}), \eqn{ca} is the
                                 ambient CO2 partial pressure in Pa (\code{ca}).

            >>> # Example taken from internals of example(rpmodel) in R implementation
        >>> vals = CalcOptimalChi(kmm = 46.09928, gammastar = 3.33925,
        ...                       ns_star = 1.12536, ca = 40.53, vpd = 1000)
        >>> vals.chi # doctest: +ELLIPSIS
        0.69435...
        >>> vals.mc # doctest: +ELLIPSIS
        0.33408...
        >>> vals.mj # doctest: +ELLIPSIS
        0.71230...
        >>> vals.mjoc # doctest: +ELLIPSIS
        2.13211...

        Args:
            kmm: Michaelis-Menten coefficient (Pa)
            gammastar:
            ns_star: viscosity correction factor for water (unitless)
            ca:
            vpd:  vapor pressure deficit (Pa) ??? array?
            c4: Use c4 representation.

        Returns:
            A dotmap containing values for ratio of ci/ca (chi, float), mc (float),
             mj (float) and mjoc (float)
    """

    def __init__(self, kmm: float, gammastar: float, ns_star: float,
                 ca: float, vpd: float, method: str = 'prentice14'):

        self.kmm = kmm
        self.gammastar = gammastar
        self.ns_star = ns_star
        self.ca = ca
        self.vpd = vpd
        self.beta = PARAM.stocker19.beta
        # return values
        self.chi = None
        self.mc = None
        self.mj = None
        self.mjoc = None

        all_methods = {'prentice14': self._prentice14, 'c4': self._c4}

        if method in all_methods:
            this_method = all_methods[method]
            this_method()
        else:
            raise ValueError(f"CalcOptimalChi: method argument '{method}' invalid.")

    def _c4(self):

        # Dummy values to represent c4 pathway
        self.chi = 1.0
        self.mc = 1.0
        self.mj = 1.0
        self.mjoc = 1.0

    def _prentice14(self):

        # TODO - original R code mentioned vectorisation but none of the inputs
        #    are obviously a vector, and the test case returns a list of scalars
        #    so it is not obvious that this is required.
    
        # Avoid negative VPD (dew conditions)
        vpd = 0 if self.vpd < 0 else self.vpd # ??? array
    
        # leaf-internal-to-ambient CO2 partial pressure (ci/ca) ratio
        xi = numpy.sqrt((self.beta * (self.kmm + self.gammastar)) / (1.6 * self.ns_star))
        self.chi = self.gammastar / self.ca + (1.0 - self.gammastar / self.ca) * xi / (xi + numpy.sqrt(vpd))
    
        # Define variable substitutes:
        vdcg = self.ca - self.gammastar
        vacg = self.ca + 2.0 * self.gammastar
        vbkg = self.beta * (self.kmm + self.gammastar)
    
        # Calculate mj, based on the mc' formulation (see Regressing_LUE.pdf)
        if self.ns_star > 0 and vpd > 0 and vbkg > 0:
            vsr = numpy.sqrt(1.6 * self.ns_star * vpd / vbkg)
            self.mj = vdcg / (vacg + 3.0 * self.gammastar * vsr)
        else:
            self.mj = None
    
        # alternative variables
        gamma = self.gammastar / self.ca
        kappa = self.kmm / self.ca
    
        # mc and mj:mv
        self.mc = (self.chi - gamma) / (self.chi + kappa)
        self.mjoc = (self.chi + kappa) / (self.chi + 2 * gamma)


class CalcLUEVcmax:

    """


    """

    def __init__(self, out_optchi, kphio, ftemp_kphio, soilmstress, method):

        self.out_optchi = out_optchi
        self.kphio = kphio
        self.ftemp_kphio = ftemp_kphio
        self.soilmstress = soilmstress
        self.lue = None
        self.vcmax_unitiabs = None
        self.omega = None
        self.omega_star = None

        all_methods = {'wang17': self._wang17, 'smith19': self._smith19,
                       'none': self._none, 'c4': self._c4}

        if method in all_methods:
            this_method = all_methods[method]
            this_method()
        else:
            raise ValueError(f"CalcLUEVcmax: method argument '{method}' invalid.")

    def _wang17(self):
        """
        >>> # Example taken from internals of example(rpmodel) in R implementation
        >>> optchi = CalcOptimalChi(kmm = 46.09928, gammastar = 3.33925, ns_star = 1.12536,
        ...                           ca = 40.53, vpd = 1000)
        >>> out = CalcLUEVcmax(optchi, kphio = 0.081785, ftemp_kphio = 0.656,
        ...                    soilmstress = 1, method='wang17')
        >>> out.lue # doctest: +ELLIPSIS
        0.25475...
        >>> out.vcmax_unitiabs # doctest: +ELLIPSIS
        0.063488...
        >>> out.omega is None # doctest: +ELLIPSIS
        True
        >>> out.omega_star is None # doctest: +ELLIPSIS
        True
        """

        # Include effect of Jmax limitation, modify mc accounting for the
        # co-limitation hypothesis after Prentice et al. (2014)
        mpi = (self.out_optchi.mj ** 2 - PARAM.wang17.c ** (2.0 / 3.0) *
               (self.out_optchi.mj ** (4.0 / 3.0)))

        mprime = numpy.sqrt(mpi) if mpi > 0 else None

        # Light use efficiency (gpp per unit absorbed light)
        self.lue = (self.kphio * self.ftemp_kphio * mprime *
                    PARAM.k.c_molmass * self.soilmstress)

        # Vcmax normalised per unit absorbed PPFD (assuming iabs=1), with Jmax limitation
        self.vcmax_unitiabs = (self.kphio * self.ftemp_kphio * self.out_optchi.mjoc *
                               mprime / self.out_optchi.mj * self.soilmstress)

    def _smith19(self):

        """
        >>> # Example taken from internals of example(rpmodel) in R implementation
        >>> optchi = CalcOptimalChi(kmm = 46.09928, gammastar = 3.33925, ns_star = 1.12536,
        ...                           ca = 40.53, vpd = 1000)
        >>> out = CalcLUEVcmax(optchi, kphio = 0.081785, ftemp_kphio = 0.656,
        ...                    soilmstress = 1, method='smith19')
        >>> out.lue # doctest: +ELLIPSIS
        0.086568...
        >>> out.vcmax_unitiabs # doctest: +ELLIPSIS
        0.021574...
        >>> out.omega # doctest: +ELLIPSIS
        1.10204...
        >>> out.omega_star # doctest: +ELLIPSIS
        1.28250...
        """

        # Adopted from Nick Smith's code:
        # Calculate omega, see Smith et al., 2019 Ecology Letters

        def _calc_omega(theta, c_cost, m):
            cm = 4 * c_cost / m  # simplification term for omega calculation
            v = 1 / (cm * (1 - theta * cm)) - 4 * theta  # simplification term for omega calculation

            # account for non-linearities at low m values
            capP = (((1 / 1.4) - 0.7) ** 2 / (1 - theta)) + 3.4
            aquad = -1
            bquad = capP
            cquad = -(capP * theta)
            m_star = (4 * c_cost) / numpy.polynomial.polynomial.polyroots([aquad, bquad, cquad])

            if m < m_star[0].real:
                return -(1 - (2 * theta)) - numpy.sqrt((1 - theta) * v)
            else:
                return -(1 - (2 * theta)) + numpy.sqrt((1 - theta) * v)

        # factors derived as in Smith et al., 2019
        self.omega = _calc_omega(theta=PARAM.smith19.theta, # Eq. S4
                                 c_cost=PARAM.smith19.c_cost,
                                 m=self.out_optchi.mj)
        self.omega_star = (1.0 + self.omega -  # Eq. 18
                           numpy.sqrt((1.0 + self.omega) ** 2 -
                                      (4.0 * PARAM.smith19.theta * self.omega)))

        # Effect of Jmax limitation
        mprime = self.out_optchi.mj * self.omega_star / (8.0 * PARAM.smith19.theta)

        # Light use efficiency (gpp per unit absorbed light)
        self.lue = (self.kphio * self.ftemp_kphio * mprime *
                    PARAM.k.c_molmass * self.soilmstress)

        # calculate Vcmax per unit aborbed light
        self.vcmax_unitiabs = (self.kphio * self.ftemp_kphio * self.out_optchi.mjoc *
                               self.omega_star / (8.0 * PARAM.smith19.theta) * self.soilmstress)  # Eq. 19


    def _none(self):
        """
        >>> # Example taken from internals of example(rpmodel) in R implementation
        >>> optchi = CalcOptimalChi(kmm = 46.09928, gammastar = 3.33925, ns_star = 1.12536,
        ...                           ca = 40.53, vpd = 1000)
        >>> out = CalcLUEVcmax(optchi, kphio = 0.081785, ftemp_kphio = 0.656,
        ...                    soilmstress = 1, method='none')
        >>> out.lue # doctest: +ELLIPSIS
        0.4589983...
        >>> out.vcmax_unitiabs # doctest: +ELLIPSIS
        0.1143898...
        >>> out.omega is None
        True
        >>> out.omega_star is None
        True
        """

        # Light use efficiency (gpp per unit absorbed light)
        self.lue = (self.kphio * self.ftemp_kphio * self.out_optchi.mj *
                    PARAM.k.c_molmass * self.soilmstress)

        # Vcmax normalised per unit absorbed PPFD (assuming iabs=1), with Jmax limitation
        self.vcmax_unitiabs = (self.kphio * self.ftemp_kphio * self.out_optchi.mjoc *
                               self.soilmstress)

    def _c4(self):
        """
        TODO - CHECK IF C4 is needed or if this is just _none with the default
               unity options from calc_optimal_chi for c4.
        >>> # Example taken from internals of example(rpmodel) in R implementation
        >>> optchi = CalcOptimalChi(kmm = 46.09928, gammastar = 3.33925, ns_star = 1.12536,
        ...                           ca = 40.53, vpd = 1000)
        >>> out = CalcLUEVcmax(optchi, kphio = 0.081785, ftemp_kphio = 0.656,
        ...                    soilmstress = 1, method='c4')
        >>> out.lue # doctest: +ELLIPSIS
        0.6443855...
        >>> out.vcmax_unitiabs # doctest: +ELLIPSIS
        0.05365096...
        >>> out.omega is None
        True
        >>> out.omega_star is None
        True
        """
        # Light use efficiency (gpp per unit absorbed light)
        self.lue = self.kphio * self.ftemp_kphio * PARAM.k.c_molmass * self.soilmstress
        # Vcmax normalised per unit absorbed PPFD (assuming iabs=1), with Jmax limitation
        self.vcmax_unitiabs = self.kphio * self.ftemp_kphio * self.soilmstress


def co2_to_ca(co2: float, patm: float) -> float:
    """Converts ca (ambient CO2) from ppm to Pa.

    Args:
        co2: annual atm. CO2, ppm
        patm: monthly atm. pressure, Pa

    Returns:
        Ambient CO2 in units of Pa
    """

    return 1.0e-6 * co2 * patm  # Pa, atms. CO2
