"""The pmodel_const module TODO."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class PModelConst(ConstantsClass):
    r"""Constants for the P Model.

    This dataclass provides the following underlying constants used in calculating the
    predictions of the P Model. Values are shown with mathematical notation, default
    value and units shown in brackets and the sources for default parameterisations are
    given below:

    * **Density of water**. Values for the Tumlirz equation taken from Table 5 of
      :cite:t:`Fisher:1975tm`:
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.fisher_dial_lambda`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.fisher_dial_Po`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.fisher_dial_Vinf`)

    * **Viscosity of water**. Values for the parameterisation taken from Table 2 and 3
      of :cite:t:`Huber:2009fy`:
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.huber_tk_ast`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.huber_rho_ast`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.huber_mu_ast`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.huber_H_i`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.huber_H_ij`)

    * **Temperature scaling of dark respiration**. Values taken from
      :cite:t:`Heskel:2016fg`:
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.heskel_b`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.heskel_c`)

    * **Temperature and entropy of VCMax**. Values taken from Table 3 of
      :cite:t:`Kattge:2007db`
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.kattge_knorr_a_ent`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.kattge_knorr_b_ent`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.kattge_knorr_Ha`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.kattge_knorr_Hd`)

    * **Scaling of Kphio with temperature**. The parameters of quadratic functions for
      the temperature dependence of Kphio are:
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.kphio_C4`, C4 plants, Eqn 5 of
      :cite:t:`cai:2020a`; and
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.kphio_C3`, C3 plants, Table 2
      of :cite:t:`Bernacchi:2003dc`.

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

    * **Soil moisture stress**. Parameterisation from :cite:t:`Stocker:2020dh`
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.soilmstress_theta0`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.soilmstress_thetastar`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.soilmstress_a`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.soilmstress_b`)

    * **Unit cost ratios (beta)**. The value for C3 plants is taken from
      :cite:t:`Stocker:2020dh`. For C4 plants, we follow the estimates of the
      :math:`g_1` parameter for C3 and C4 plants in :cite:t:`Lin:2015wh`  and
      :cite:t:`DeKauwe:2015im`, which have a C3/C4 ratio of around 3. Given that
      :math:`g_1 \equiv \xi \propto \surd\beta`, a reasonable default for C4 plants is
      that :math:`\beta_{C4} \approx \beta_{C3} / 9 \approx 146 /  9 \approx 16.222`.
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_prentice14`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`)

    * **Unit cost ratios (beta) response to soil moisture**. These constants set the
      response of beta to soil moisture for the
      :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.lavergne20_c3` method and for
      :meth:`~pyrealm.pmodel.pmodel.CalcOptimalChi.lavergne20_c4`.
      (:attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_b_c3`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_a_c3`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_b_c4`,
      :attr:`~pyrealm.constants.pmodel_const.PModelConst.lavergne_2020_a_c4`)

    * **Electron transport capacity maintenance cost** Value taken from
      :cite:t:`Wang:2017go`

    * **Calculation of omega**. Values for estimating the scaling factor in J max
      limitation method of :cite:t:`Smith:2019dv`.

    * **Dark respiration**. Value taken from :cite:t:`Atkin:2015hk` for C3 herbaceous
      plants

    """

    # Constants
    k_R: float = 8.3145
    """Universal gas constant (:math:`R` , 8.3145, J/mol/K)"""
    k_co: float = 209476.0
    """O2 partial pressure, Standard Atmosphere (:math:`co` , 209476.0, ppm)"""
    k_c_molmass: float = 12.0107
    """Molecular mass of carbon (:math:`c_molmass` , 12.0107, g)"""
    k_Po: float = 101325.0
    """Standard reference atmosphere (Allen, 1973)   (:math:`P_o` , 101325.0, Pa)"""
    k_To: float = 25.0
    """Standard reference temperature (Prentice, unpublished) (:math:`T_o` , 25.0,
    °C)"""
    k_L: float = 0.0065
    """Adiabiatic temperature lapse rate (Allen, 1973)   (:math:`L` , 0.0065, K/m)"""
    k_G: float = 9.80665
    """Gravitational acceleration (:math:`G` , 9.80665, m/s^2)"""
    k_Ma: float = 0.028963
    """Molecular weight of dry air (Tsilingiris, 2008)  (:math:`M_a`, 0.028963,
    kg/mol)"""
    k_CtoK: float = 273.15
    """Conversion from °C to K   (:math:`CtoK` , 273.15, -)"""

    # Fisher Dial
    fisher_dial_lambda: NDArray[np.float32] = np.array(
        [1788.316, 21.55053, -0.4695911, 0.003096363, -7.341182e-06]
    )
    r"""Temperature dependent lambda parameterisation of the Tumlirz equation.
     (:math:`\lambda`)."""

    fisher_dial_Po: NDArray[np.float32] = np.array(
        [5918.499, 58.05267, -1.1253317, 0.0066123869, -1.4661625e-05]
    )
    """Temperature dependent P0 parameterisation of the Tumlirz equation
    (:math:`P_0`)."""

    fisher_dial_Vinf: NDArray[np.float32] = np.array(
        [
            0.6980547,
            -0.0007435626,
            3.704258e-05,
            -6.315724e-07,
            9.829576e-09,
            -1.197269e-10,
            1.005461e-12,
            -5.437898e-15,
            1.69946e-17,
            -2.295063e-20,
        ]
    )
    r"""Temperature dependent Vinf parameterisation of the Tumlirz equation
    (:math:`V_{\infty}`)."""

    # Huber
    simple_viscosity: bool = False
    """Boolean setting for use of simple viscosity calculations"""
    huber_tk_ast: float = 647.096
    """Huber reference temperature (:math:`tk_{ast}`, 647.096, Kelvin)"""
    huber_rho_ast: float = 322.0
    r"""Huber reference density (:math:`\rho_{ast}`, 322.0, kg/m^3)"""
    huber_mu_ast: float = 1e-06
    r"""Huber reference pressure (:math:`\mu_{ast}` 1.0e-6, Pa s)"""

    huber_H_i: NDArray[np.float32] = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
    """Temperature dependent parameterisation of Hi in Huber."""
    huber_H_ij: NDArray[np.float32] = np.array(
        [
            [0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
            [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
            [-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0],
            [0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0],
            [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
            [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264],
        ],
    )
    """Temperature and mass density dependent parameterisation of Hij in Huber."""

    # Heskel
    heskel_b: float = 0.1012
    """Linear coefficient of scaling of dark respiration (:math:`b`, 0.1012)"""
    heskel_c: float = 0.0005
    """Quadratic coefficient of scaling of dark respiration (:math:`c`, 0.0005)"""

    # KattgeKnorr
    kattge_knorr_a_ent: float = 668.39
    """Offset of entropy vs. temperature relationship (:math:`a_{ent}`, 668.39,
    J/mol/K)"""
    kattge_knorr_b_ent: float = -1.07
    """Slope of entropy vs. temperature relationship (:math:`b_{ent}`, -1.07,
    J/mol/K^2)"""
    kattge_knorr_Ha: float = 71513
    """Activation energy (:math:`H_a`, 71513, J/mol)"""
    kattge_knorr_Hd: float = 200000
    """Deactivation energy (:math:`H_d`, 200000, J/mol)"""

    # Kphio:
    # - note that kphio_C4 has been updated to account for an unintended double
    #   8 fold downscaling to account for the fraction of light reaching PS2.
    #   from original values of [-0.008, 0.00375, -0.58e-4]
    kphio_C4: NDArray[np.float32] = np.array((-0.064, 0.03, -0.000464))
    """Quadratic scaling of Kphio with temperature for C4 plants"""
    kphio_C3: NDArray[np.float32] = np.array((0.352, 0.022, -0.00034))
    """Quadratic scaling of Kphio with temperature for C3 plants"""

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

    # Soilmstress
    soilmstress_theta0: float = 0.0
    """Lower bound in relative soil moisture"""
    soilmstress_thetastar: float = 0.6
    """Upper bound in relative soil moisture"""
    soilmstress_a: float = 0.0
    """Intercept of aridity sensitivity function for soil moisture"""
    soilmstress_b: float = 0.733
    """Slope of aridity sensitivity function for soil moisture"""

    # Unit cost ratio (beta) values for different CalcOptimalChi methods
    beta_cost_ratio_prentice14: NDArray[np.float32] = np.array([146.0])
    r"""Unit cost ratio for C3 plants (:math:`\beta`, 146.0)."""
    beta_cost_ratio_c4: NDArray[np.float32] = np.array([146.0 / 9])
    r"""Unit cost ratio for C4 plants (:math:`\beta`, 16.222)."""
    lavergne_2020_b_c3: float = 1.73
    """Slope of soil moisture effects on beta for C3 plants"""
    lavergne_2020_a_c3: float = 4.55
    """Intercept of soil moisture effects on beta for C3 plants"""
    lavergne_2020_b_c4: float = 1.73
    """Slope of soil moisture effects on beta for C4 plants"""
    lavergne_2020_a_c4: float = 4.55 - np.log(9)
    """Slope of soil moisture effects on beta for C4 plants"""

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
