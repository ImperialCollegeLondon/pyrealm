"""The pmodel_const module TODO."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class PModelConst(ConstantsClass):
    r"""Constants for the P Model.

    This dataclass provides the following underlying constants used in calculating the
    predictions of the P Model.



    * **Enzyme kinetics for VCMax**. Values taken from Table 3 of
      :cite:t:`Kattge:2007db`.



    * **Temperature responses of photosynthetic enzymes**. Values taken from Table 1 of
      :cite:t:`Bernacchi:2001kg`. `kc_25` and `ko_25` are converted from µmol mol-1 and
      mmol mol-1, assuming a measurement at an elevation of 227.076 metres and standard
      atmospheric pressure for that elevation (98716.403 Pa).
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.bernacchi_dhac`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.bernacchi_dhao`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.bernacchi_dha`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.bernacchi_kc25`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.bernacchi_ko25`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.bernacchi_gs25_0`)



    * **Unit cost ratios (beta)**. The value for C3 plants is taken from
      :cite:t:`Stocker:2020dh`. For C4 plants, we follow the estimates of the
      :math:`g_1` parameter for C3 and C4 plants in :cite:t:`Lin:2015wh`  and
      :cite:t:`DeKauwe:2015im`, which have a C3/C4 ratio of around 3. Given that
      :math:`g_1 \equiv \xi \propto \surd\beta`, a reasonable default for C4 plants is
      that :math:`\beta_{C4} \approx \beta_{C3} / 9 \approx 146 /  9 \approx 16.222`.
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_prentice14`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`)



    * **Electron transport capacity maintenance cost** Value taken from
      :cite:t:`Wang:2017go`

    * **Calculation of omega**. Values for estimating the scaling factor in J max
      limitation method of :cite:t:`Smith:2019dv`.

    * **Dark respiration**. Value taken from :cite:t:`Atkin:2015hk` for C3 herbaceous
      plants

    """

    sandoval_peak_phio: tuple[float, float] = (6.8681, 0.07956432)
    """Curvature parameters for calculation of peak phio in the Sandoval method for
    estimation of quantum yield efficiency."""

    sandoval_kinetics: tuple[float, float, float, float] = (
        1558.853,
        -50.223,
        294.804,
        75000.0,
    )
    """Enzyme kinetics parameters for estimation of kphio from mean growth temperature
    in the Sandoval method :cite:t:`sandoval:in_prep` for estimation of quantum yield
    efficiency. Values are: the intercept and slope of activation entropy as a function
    of the mean growth temperature (J/mol/K), the deactivation energy constant (J/mol)
    and the activation energy (J/mol). """

    plant_T_ref: float = 25.0
    """Standard baseline reference temperature of photosynthetic processes (Prentice,
    unpublished)  (:math:`T_o` , 25.0, °C)"""

    # Heskel
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
    """Coefficients of Arrhenius factor scaling for :math:`V_{cmax}`. The `simple`
    method provides an estimate of the activation energy (:math:`H_a`, J/mol). The
    `kattge_knorr` method provides the parameterisation from :cite:t:`Kattge:2007db`, 
    providing the intercept and slope of activation entropy as a function of the mean
    growth temperature (J/mol/K), the deactivation energy constant (:math:`H_d`, J/mol) 
    and the activation energy (J/mol). (:math:`H_a`, J/mol)."""
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
    """Coefficients of Arrhenius factor scaling for :math:`J_{max}`. The `simple`
    method provides an estimate of the activation energy (:math:`H_a`, J/mol). The
    `kattge_knorr` method provides the parameterisation from :cite:t:`Kattge:2007db`, 
    providing the intercept and slope of activation entropy as a function of the mean
    growth temperature (J/mol/K), the deactivation energy constant (:math:`H_d`, J/mol) 
    and the activation energy (J/mol). (:math:`H_a`, J/mol)."""

    kphio_C4: tuple[float, float, float] = (-0.064, 0.03, -0.000464)
    """Coefficients of the quadratic scaling of the quantum yield of photosynthesis
    (``phi_0``, :math:`\phi_0`) with temperature for C4 plants, taken from Eqn 5 of
    :cite:t:`cai:2020a`, and adjusted from the original values (-0.008, 0.00375,
    -0.58e-4) to account for an unintentional double scaling to account for the fraction
    of light reaching PS2."""

    kphio_C3: tuple[float, float, float] = (0.352, 0.022, -0.00034)
    """Coefficients of the quadratic scaling of the quantum yield of photosynthesis
    (``phi_0``, :math:`\phi_0`) with temperature for C3 plants, taken from Table 2 of
    :cite:t:`Bernacchi:2003dc`"""

    # Bernachhi
    bernacchi_dhac: float = 79430
    """Bernacchi estimate of activation energy Kc for CO2 (J/mol)"""
    bernacchi_dhao: float = 36380
    """Bernacchi estimate of activation energy Ko for O2 (J/mol)"""
    bernacchi_dha: float = 37830
    """Bernacchi estimate of activation energy for gammastar (J/mol)"""
    bernacchi_kc25: float = 39.97  # Reported as 404.9 µmol mol-1
    """Bernacchi estimate of kc25"""
    bernacchi_ko25: float = 27480  # Reported as 278.4 mmol mol-1
    """Bernacchi estimate of ko25"""
    bernacchi_gs25_0: float = 4.332  # Reported as 42.75 µmol mol-1
    """Bernacchi estimate of gs25_0"""

    # Boyd
    boyd_kp25_c4: float = 16  # Pa  from Boyd et al. (2015)
    boyd_dhac_c4: float = 36300  # J mol-1
    # boyd_dhac_c4: float = 79430
    # boyd_dhao_c4: float = 36380
    # boyd_dha_c4: float = 37830
    # boyd_kc25_c4: float = 41.03
    # boyd_ko25_c4: float = 28210
    # boyd_gs25_0_c4: float = 2.6

    soilmstress_stocker: dict[str, float] = field(
        default_factory=lambda: dict(theta0=0, thetastar=0.6, a=0.0, b=0.733)
    )
    """Parameterisation of the soil moisture stress function of
    :cite:t:`Stocker:2020dh` as a dictionary providing values for the:
    
    * Intercept of the aridity sensitivity function (``theta0``),
    * Slope of the aridity sensitivity function (``thetastar``),
    * Lower bound in relative soil moisture  (``a``), and
    * Upper bound in relative soil moisture (``b``).
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

    # Unit cost ratio (beta) values for different CalcOptimalChi methods
    beta_cost_ratio_prentice14: NDArray[np.float64] = field(
        default_factory=lambda: np.array([146.0])
    )
    r"""Unit cost ratio for C3 plants (:math:`\beta`, 146.0)."""
    beta_cost_ratio_c4: NDArray[np.float64] = field(
        default_factory=lambda: np.array([146.0 / 9])
    )
    r"""Unit cost ratio for C4 plants (:math:`\beta`, 16.222)."""

    lavergne_2020_c3: tuple[float, float] = (4.55, 1.73)
    """Intercept and slope coefficients for the effects of soil moisture on optimal chi
    estimates for C3 plants, following :cite:`lavergne:2020a`."""

    lavergne_2020_c4: tuple[float, float] = (4.55 - np.log(9), 1.73)
    """Intercept and slope coefficients for the effects of soil moisture on optimal chi
    estimates for C4 plants, following :cite:`lavergne:2020a`."""

    # Wang17
    wang17_c: float = 0.41
    """Unit carbon cost for the maintenance of electron transport capacity (:math:`c`,
    0.41, )"""

    # Smith19
    smith19_theta: float = 0.85
    r"""Scaling factor theta for Jmax limitation (:math:`\theta`, 0.85)"""
    smith19_c_cost: float = 0.05336251
    r"""Scaling factor c for Jmax limitation (:math:`c`, 0.05336251)"""

    # Atkin
    atkin_rd_to_vcmax: float = 0.015
    """Ratio of Rdark to Vcmax25 (0.015)"""
