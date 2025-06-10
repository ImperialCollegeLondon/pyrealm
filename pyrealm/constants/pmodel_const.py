"""The pmodel_const module TODO."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class PModelConst(ConstantsClass):
    r"""Constants for the P Model module.

    This dataclass provides constants used in calculating the predictions of the P Model
    and associated methods.
    """

    sandoval_peak_phio: tuple[float, float] = (6.8681, 0.07956432)
    """Curvature parameters for calculation of peak phio in the Sandoval method for
    estimation of quantum yield efficiency."""

    sandoval_kinetics: dict[str, float] = field(
        default_factory=lambda: dict(
            entropy_intercept=1558.853,
            entropy_slope=-50.223,
            ha=75000.0,
            hd=294.804,
        )
    )
    """Enzyme kinetics parameters for estimation of kphio from mean growth temperature
    in the Sandoval method :cite:t:`sandoval:in_prep` for estimation of quantum yield
    efficiency. Values are: the intercept and slope of activation entropy as a function
    of the mean growth temperature (J/mol/K), the deactivation energy constant (J/mol)
    and the activation energy (J/mol). """

    tc_ref: float = 25.0
    """Standard baseline reference temperature of photosynthetic processes in °C 
    (:math:`T_o` , 25.0, °C)"""

    tk_ref: float = 298.15
    """Standard baseline reference temperature of photosynthetic processes in Kelvin 
    (298.15, K)"""

    # NOTE that tk_ref might be a considered a duplication of CoreConst.k_To but that
    # refers explicitly to the physical standard temperature and this is a reference
    # temperature for photosynthesis. It is _vanishingly_ unlikely that this will
    # change but they could _in theory_ be different.

    heskel_rd: tuple[float, float] = (0.1012, 0.0005)
    """Linear (:math:`b`, 0.1012) and quadratic  (:math:`c`, 0.0005) coefficients of the
    temperature scaling of dark respiration. Values taken from
    :cite:t:`Heskel:2016fg`:."""

    # Arrhenius values
    arrhenius_vcmax: dict = field(
        default_factory=lambda: dict(
            simple=dict(ha=65330),
            kattge_knorr=dict(
                entropy_intercept=668.39,
                entropy_slope=-1.07,
                ha=71513,
                hd=200000,
            ),
        )
    )
    """Coefficients of Arrhenius factor scaling for :math:`V_{cmax}`. The ``simple``
    method provides an estimate of the activation energy (:math:`H_a`, J/mol). The
    ``kattge_knorr`` method provides the parameterisation from :cite:t:`Kattge:2007db`, 
    providing the intercept and slope of activation entropy as a function of the mean
    growth temperature (J/mol/K), the deactivation energy constant (:math:`H_d`, J/mol) 
    and the activation energy (:math:`H_a`, J/mol)."""

    arrhenius_jmax: dict = field(
        default_factory=lambda: dict(
            simple=dict(ha=43900),
            kattge_knorr=dict(
                entropy_intercept=659.70,
                entropy_slope=-0.75,
                ha=49884,
                hd=200000,
            ),
        )
    )
    """Coefficients of Arrhenius factor scaling for :math:`J_{max}`. The ``simple``
    method provides an estimate of the activation energy (:math:`H_a`, J/mol). The
    ``kattge_knorr`` method provides the parameterisation from :cite:t:`Kattge:2007db`, 
    providing the intercept and slope of activation entropy as a function of the mean
    growth temperature (J/mol/K), the deactivation energy constant (:math:`H_d`, J/mol) 
    and the activation energy (:math:`H_a`, J/mol)."""

    kphio_C4: tuple[float, float, float] = (-0.064, 0.03, -0.000464)
    r"""Coefficients of the quadratic scaling of the quantum yield of photosynthesis
    (``phi_0``, :math:`\phi_0`) with temperature for C4 plants, taken from Eqn 5 of
    :cite:t:`cai:2020a`, and adjusted from the original values (-0.008, 0.00375,
    -0.58e-4) to account for an unintentional double scaling to account for the fraction
    of light reaching PS2."""

    kphio_C3: tuple[float, float, float] = (0.352, 0.022, -0.00034)
    r"""Coefficients of the quadratic scaling of the quantum yield of photosynthesis
    (``phi_0``, :math:`\phi_0`) with temperature for C3 plants, taken from Table 2 of
    :cite:t:`Bernacchi:2003dc`"""

    # Bernachhi
    bernacchi_kmm: dict[str, float] = field(
        default_factory=lambda: dict(
            dhac=79430.0,
            dhao=36380.0,
            kc25=39.97,
            ko25=27480.0,
        )
    )
    r"""Coefficients for the estimation of the Michaelis Menten coefficient of
    Rubisco-limited assimilation. Values are taken from Table 1 of
    :cite:t:`Bernacchi:2001kg` and provide a dictionary of:
    
    * The activation energy for :math:`\ce{CO2}` (:math:`\Delta H_{kc}`, ``dhac``,
      J/mol) 
    * The activation energy for :math:`\ce{O2}` (:math:`\Delta H_{ko}`, `dhao``, J/mol))
    * The Michelis constant for :math:`\ce{CO2}` at standard temperature
      (:math:`K_{c25}`, ``kc25``, Pa) 
    * The Michelis constant for :math:`\ce{O2}` at standard temperature
      (:math:`K_{o25}`, ``ko25``, Pa)

    The values for  `kc_25` and `ko_25` are converted from the original tabulated values
    of 404.9 µmol mol-1 and 278.4 mmol mol-1 respectively to Pa, assuming a measurement
    at an elevation of 227.076 metres and standard  atmospheric pressure for that
    elevation (98716.403 Pa).
    """

    maximum_phi0: float = 1.0 / 8.0
    r"""The theoretical maximum intrinsic quantum yield efficiency of photosynthesis
    (:math:`\phi_0`, unitless). This value defaults to 1/8 but could be set to 1/9 to
    capture the absence of a Q cycle :cite:`long:1993a`."""

    bernacchi_gs: dict[str, float] = field(
        default_factory=lambda: dict(
            dha=37830.0,
            gs25_0=4.332,
        )
    )
    r"""Coefficients for the estimation of the photorespiratory CO2 compensation point.
    Values are taken from Table 1 of :cite:t:`Bernacchi:2001kg` and provide a dictionary
    of: 
    
    * The reference value of :math:`\Gamma^{*}` at standard temperature and pressure
      (:math:`\Gamma^{*}_{0}`, ``gs25_0``, Pa), converted from the tabulated value of
      42.75 µmol mol-1, , assuming a measurement at an elevation of 227.076 metres and
      standard  atmospheric pressure for that elevation (98716.403 Pa).
    * The activation energy (:math:`\Delta H_a`, ``dha``, J/mol)

    The values for  `kc_25` and `ko_25` are converted from the original tabulated values
    of 404.9 µmol mol-1 and 278.4 mmol mol-1 respectively to Pa, assuming a measurement
    at an elevation of 227.076 metres and standard  atmospheric pressure for that
    elevation (98716.403 Pa).
    """

    soilmstress_stocker: dict[str, float] = field(
        default_factory=lambda: dict(theta0=0, thetastar=0.6, a=0.0, b=0.733)
    )
    """Parameterisation of the soil moisture stress function of
    :cite:t:`Stocker:2020dh` as a dictionary providing values for the: 
    
    * intercept of the aridity sensitivity function (``theta0``),
    * slope of the aridity sensitivity function (``thetastar``),
    * lower bound in relative soil moisture  (``a``), and
    * upper bound in relative soil moisture (``b``).
    """

    soilmstress_mengoli: dict[str, float] = field(
        default_factory=lambda: dict(psi_a=0.34, psi_b=-0.6, y_a=0.62, y_b=-0.45)
    )
    """Parameterisation of the soil moisture stress function of
    :cite:t:`mengoli:2023a` as a dictionary providing values for the:

    * coefficient of the maximal level function  (``y_a``),
    * exponent of the maximal level function  (``y_b``),
    * coefficient of the threshold function  (``psi_a``), and
    * exponent of the threshold function (``psi_b``).
    """

    beta_cost_ratio_c3: NDArray[np.float64] = field(
        default_factory=lambda: np.array([146.0])
    )
    r"""Unit cost ratio for C3 plants (:math:`\beta`, 146.0) taken from
     :cite:t:`Stocker:2020dh`."""

    beta_cost_ratio_c4: NDArray[np.float64] = field(
        default_factory=lambda: np.array([146.0 / 9])
    )
    r"""Unit cost ratio for C4 plants (:math:`\beta`, 16.222).
    
    For C4 plants, we follow the estimates of the   :math:`g_1` parameter for C3 and C4
    plants in :cite:t:`Lin:2015wh`  and :cite:t:`DeKauwe:2015im`, which have a C3/C4
    ratio of around 3. Given that :math:`g_1 \equiv \xi \propto \surd\beta`, a
    reasonable default for C4 plants is that :math:`\beta_{C4} \approx \beta_{C3} / 9
    \approx 146 /  9 \approx 16.222`."""

    lavergne_2020_c3: tuple[float, float] = (4.55, 1.73)
    """Intercept and slope coefficients for the effects of soil moisture on optimal chi
    estimates for C3 plants, following :cite:`lavergne:2020a`."""

    lavergne_2020_c4: tuple[float, float] = (4.55 - np.log(9), 1.73)
    """Intercept and slope coefficients for the effects of soil moisture on optimal chi
    estimates for C4 plants, following :cite:`lavergne:2020a`."""

    wang17_c: float = 0.41
    """Unit carbon cost for the maintenance of electron transport capacity, value taken
    from  :cite:t:`Wang:2017go` (:math:`c`, 0.41, )."""

    # Smith19
    smith19_coef: tuple[float, float] = (0.85, 0.05336251)
    r"""Scaling factors ``theta`` (:math:`\theta`, 0.85) and ``c`` (:math:`c`,
    0.05336251) used in the calculation of J_max limitation using the method of 
    :cite:t:`Smith:2019dv`."""
