"""The param_classes module.

Many of the functions and classes in `pyrealm` have an underlying set of
parameters that will typically be held constant. This includes true universal
constants, such as the molar mass of Carbon, but also a large number of
calculated parameters from the literature.

These underlying parameters are not hardcoded, but rather parameter classes are
defined for different models within `pyrealm`. These parameter configurations
can be altered and can also be loaded and saved to configuration files. The
{class}`~pyrealm.param_classes.ParamClass` base class provides the basic load
and save methods and then individual parameter classes define the parameter sets
and default values for each class.


This implementation has the following desired features:

1. Ability to use a obj.attr notation rather than obj['attr'].
2. Ability to freeze values to avoid them being edited in use - distinctly paranoid!
3. Ability to set a default mapping with default values/
4. Typing to set expected types on default values.
5. Simple export/import methods to go to from dict / JSON
6. Is a class, to allow __repr__ and other methods.

... and then there is a tricky one:

7. Extensibility. This is the hard one and currently would only be needed
   to support a customisable version of the T Model. If the T Model could
   have overridden geometry methods, then these settings _have_ to be able
   to take extra parameters. And having to set a type on those is another
   thing that users aren't going to buy into? This makes using @dataclass
   tricky - because extending class attributes on the fly is really not
   something that comes naturally to a class. A dotted dict replacement,
   like Box or addict, is a fairly simpl functional swap.

"""


import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from dacite import from_dict
from numpy.typing import NDArray


class ParamClass:
    """Base class for model parameter classes.

    This base class provides a consistent interface for creating and exporting
    data from model parameter classes. It defines methods to create parameter
    instances from a dictionary of values and from a JSON file of values.
    It also defines methods to export an existing instance of a parameter
    class to a dictionary or a JSON file.

    All model parameter classes use the :class:`~dataclasses.dataclass`
    class as the basic structure. All fields in these classes are defined
    with default values, so instances can be defined using partial dictionaries
    of alternative values.
    """

    def to_dict(self) -> dict:
        """Return a parameter dictionary.

        This method returns the attributes and values used in an instance of a parameter
        class object as a dictionary.

        Returns:
            A dictionary
        """
        return asdict(self)

    def to_json(self, filename: str) -> None:
        """Export a parameter set to JSON.

        This method writes the attributes and values used in an instance of a
        parameter class object to a JSON file.

        Args:
            filename: A path to the JSON file to be created.

        Returns:
            None
        """
        with open(filename, "w") as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

    @classmethod
    def from_dict(cls, data: dict) -> "ParamClass":
        """Create a ParamClass instance from a dictionary.

        Generates a parameter class object using the data provided in a
        dictionary to override default values

        Args:
            data: A dictionary, keyed by parameter class attribute names,
                  providing values to use in a new instance.

        Returns:
            An instance of the parameter class.
        """
        return from_dict(cls, data)

    @classmethod
    def from_json(cls, filename: str) -> "ParamClass":
        """Create a ParamClass instance from a JSON file.

        Generates a parameter class object using the data provided in a
        JSON file.

        Args:
            filename: The path to a JSON formatted file containing a set
                     of values keyed by parameter class attribute names
                     to use in a new instance.

        Returns:
            An instance of the parameter class.
        """
        with open(filename, "r") as infile:
            json_data = json.load(infile)

        return cls.from_dict(json_data)


@dataclass(frozen=True)
class PModelParams(ParamClass):
    r"""Model parameters for the P Model.

    This dataclass provides a large set of underlying parameters used in
    calculating the predictions of the P Model. The traits are shown below
    with mathematical notation, default value and units shown in brackets:

    **True constants**

    * `k_R`: Universal gas constant (:math:`R` , 8.3145, J/mol/K)
    * `k_co`: O2 partial pressure, Standard Atmosphere (:math:`co` , 209476.0, ppm)
    * `k_c_molmass`: Molecular mass of carbon (:math:`c_molmass` , 12.0107, g)
    * `k_Po`: Standard atmosphere (Allen, 1973)   (:math:`P_o` , 101325.0, Pa)
    * `k_To`: Reference temperature (Prentice, unpublished)   (:math:`T_o` , 25.0, °C)
    * `k_L`: Adiabiatic temperature lapse rate (Allen, 1973)   (:math:`L` , 0.0065, K/m)
    * `k_G`: Gravitational acceleration (:math:`G` , 9.80665, m/s^2)
    * `k_Ma`: Molecular weight of dry air (Tsilingiris, 2008)  (:math:`M_a`,
       0.028963, kg/mol)
    * `l_CtoK`: Conversion from °C to K   (:math:`CtoK` , 273.15, -)

    ** Density of water**, values taken from Table 5 of :cite:`Fisher:1975tm`.

    * `fisher_dial_lambda`: [1788.316, 21.55053, -0.4695911, 3.096363e-3, -7.341182e-6]
    * `fisher_dial_Po`: [5918.499, 58.05267, -1.1253317, 6.613869e-3, -1.4661625e-5]
    * `fisher_dial_Vinf`: [0.6980547, -7.435626e-4, 3.704258e-5, -6.315724e-7,
      9.829576e-9, -1.197269e-10, 1.005461e-12, -5.437898e-15,
      1.69946e-17, -2.295063e-20]

    ** Viscosity of water**, values taken from :cite:`Huber:2009fy`

    * `simple_viscosity`: Use a simple implementation of viscosity (boolean)
    * `huber_tk_ast`: Reference temperature (:math:`tk_{ast}`, 647.096, Kelvin)
    * `huber_rho_ast`: Reference density (:math:`\rho_{ast}`, 322.0, kg/m^3)
    * `huber_mu_ast`: Reference pressure (:math:`\mu_{ast}` 1.0e-6, Pa s)
    * `huber_H_i`: Values of H_i (Table 2):
        (:math:`H_i`, [1.67752, 2.20462, 0.6366564, -0.241605])
    * `huber_H_ij`: Values of H_ij (Table 3):
        (:math:`H_ij`, [[0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
        [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
        [-0.281378, -0.906851, -0.772479, -0.489837, -0.257040, 0.0],
        [0.161913,  0.257399, 0.0, 0.0, 0.0, 0.0],
        [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
        [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264]])

    ** Temperature scaling of dark respiration**, values taken from
    :cite:`Heskel:2016fg`

    * `heskel_b`: Linear factor in scaling (:math:`b`, 0.1012)
    * `heskel_c`: Quadratic factor in scaling (:math:`c`, 0.0005)

    ** Temperature and entropy of VCMax**, values taken from Table 3 of
    :cite:`Kattge:2007db`

    * `kattge_knorr_a_ent`: Offset of entropy vs. temperature relationship
        (:math:`a_{ent}`, 668.39, J/mol/K)
    * `kattge_knorr_b_ent`: Slope of entropy vs. temperature relationship
        (:math:`b_{ent}`, -1.07, J/mol/K^2)
    * `kattge_knorr_Ha`: Activation energy (:math:`H_a`, 71513, J/mol)
    * `kattge_knorr_Hd`: Deactivation energy (:math:`H_d`, 200000, J/mol)

    ** Scaling of Kphio with temperature**, parameters of quadratic functions

    * `kphio_C4`: Scaling of Kphio in C4 plants, Eqn 5 of :cite:`cai:2020a`
    * `kphio_C3`: Scaling of Kphio in C3 plants, taken from Table 2 of
      :cite:`Bernacchi:2003dc` ([0.352, 0.022, -3.4e-4])

    ** Temperature responses of photosynthetic enzymes**, values taken from Table 1
    of :cite:`Bernacchi:2003dc`. `kc_25` and `ko_25` are converted from µmol
    mol-1 and mmol mol-1, assuming a measurement at and elevation of 227.076
    metres and standard atmospheric pressure (98716.403 Pa).

    * dhac: 79430  # (J/mol) Activation energy (Kc)
    * dhao: 36380  # (J/mol) Activation energy (Ko)
    * dha: 37830  # (J/mol) Activation energy (gamma*)
    * kc25: 39.97  # Reported as 404.9 µmol mol-1
    * ko25: 27480  # Reported as 278.4 mmol mol-1
    * gs25_0: 4.332  # Reported as 42.75 µmol mol-1

    ** Soil moisture stress**, parameterisation from :cite:`Stocker:2020dh`

    * `soilmstress_theta0`: 0.0
    * `soilmstress_thetastar`: 0.6
    * `soilmstress_a`: 0.0
    * `soilmstress_b`: 0.733

    ** Unit cost ratios**, value taken from :cite:`Stocker:2020dh`

    * `stocker19_beta_c3`: Unit cost ratio for C3 plants.
        (:math:`\beta`, 146.0)
    * `stocker19_beta_c4`: Unit cost ratio for C4 plants.
        (:math:`\beta`, 146.0 / 9 = 16.2222)

    ** Electron transport capacity maintenance cost**, value taken from
    :cite:`Wang:2017go`

    *  `wang_c`: unit carbon cost for the maintenance of electron transport
        capacity (:math:`c`, 0.41, )

    ** Calculation of omega**, scaling factor in J max limitation method
    of :cite:`Smith:2019dv`:

    * `smith19_theta`: (:math:`\theta`, 0.85)
    * `smith19_c_cost`: (:math:`\c`, 0.05336251)

    ** Dark respiration**, values taken from :cite:`Atkin:2015hk` for
    C3 herbaceous plants:

    * `atkin_rd_to_vcmax`:  Ratio of Rdark to Vcmax25 (0.015)
    """

    # TODO: - look how to autodoc the descriptions from the code?
    # https://github.com/tox-dev/sphinx-autodoc-typehints/issues/44

    # Constants
    k_R: float = 8.3145
    k_co: float = 209476.0
    k_c_molmass: float = 12.0107
    k_Po: float = 101325.0
    k_To: float = 25.0
    k_L: float = 0.0065
    k_G: float = 9.80665
    k_Ma: float = 0.028963
    k_CtoK: float = 273.15

    # Fisher Dial
    fisher_dial_lambda: NDArray[np.float32] = np.array(
        [1788.316, 21.55053, -0.4695911, 0.003096363, -7.341182e-06]
    )
    fisher_dial_Po: NDArray[np.float32] = np.array(
        [5918.499, 58.05267, -1.1253317, 0.0066123869, -1.4661625e-05]
    )
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
    # Huber
    simple_viscosity: bool = False
    huber_tk_ast: float = 647.096
    huber_rho_ast: float = 322.0
    huber_mu_ast: float = 1e-06
    huber_H_i: NDArray[np.float32] = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
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
    # Heskel
    heskel_b: float = 0.1012
    heskel_c: float = 0.0005

    # KattgeKnorr
    kattge_knorr_a_ent: float = 668.39
    kattge_knorr_b_ent: float = -1.07
    kattge_knorr_Ha: float = 71513
    kattge_knorr_Hd: float = 200000

    # Kphio:
    # - note that kphio_C4 has been updated to account for an unintended double
    #   8 fold downscaling to account for the fraction of light reaching PS2.
    #   from original values of [-0.008, 0.00375, -0.58e-4]
    kphio_C4: NDArray[np.float32] = np.array((-0.064, 0.03, -0.000464))
    kphio_C3: NDArray[np.float32] = np.array((0.352, 0.022, -0.00034))

    # Bernachhi
    bernacchi_dhac: float = 79430
    bernacchi_dhao: float = 36380
    bernacchi_dha: float = 37830
    bernacchi_kc25: float = 39.97
    bernacchi_ko25: float = 27480
    bernacchi_gs25_0: float = 4.332

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
    soilmstress_thetastar: float = 0.6
    soilmstress_a: float = 0.0
    soilmstress_b: float = 0.733

    # Unit cost ratio (beta) values for different CalcOptimalChi methods
    beta_cost_ratio_prentice14: float = 146.0
    beta_cost_ratio_c4: float = 146.0 / 9
    lavergne_2020_b_c3: float = 1.73
    lavergne_2020_a_c3: float = 4.55
    lavergne_2020_b_c4: float = 1.73
    lavergne_2020_a_c4: float = 4.55 - np.log(9)

    # Wang17
    wang17_c: float = 0.41

    # Smith19
    smith19_theta: float = 0.85
    smith19_c_cost: float = 0.05336251

    # Atkin
    atkin_rd_to_vcmax: float = 0.015


@dataclass(frozen=True)
class IsotopesParams(ParamClass):
    """Settings for calculate carbon isotope discrimination.

    This data class provides values for underlying parameters used in the
    calculation of carbon isotope discrimination from P Model instances.

    The parameters are:

    """

    # Lavergne (2020)
    lavergne_delta13_a = 13.95
    lavergne_delta13_b = -17.04

    # Farquhar et al. (1982)
    farquhar_a: float = 4.4
    farquhar_b: float = 29
    farquhar_b2: float = 28
    farquhar_f: float = 12

    # vonCaemmerer et al. (2014)
    vonCaemmerer_b4: float = -7.4
    vonCaemmerer_s: float = 1.8
    vonCaemmerer_phi: float = 0.5

    # Frank et al. (2015): post-photosynthetic fractionation
    # between leaf organic matter and alpha-cellulose: 2.1 +/- 1.2 ‰
    frank_postfrac: float = 2.1

    # Badeck et al. (2005): post-photosynthetic fractionation
    # between leaf organic matter and bulk wood
    badeck_postfrac: float = 1.9


@dataclass(frozen=True)
class C3C4Params(ParamClass):
    r"""Model parameters for the C3C4Competition class.

    This data class holds statistical estimates used to calculate the fraction
    of C4 plants based on the relative GPP of C3 and C4 plants for given
    conditions and estimated treecover.
    """

    # Non-linear regression of fraction C4 plants from proportion GPP advantage
    # of C4 over C3 plants
    adv_to_frac_k = 6.63
    adv_to_frac_q = 0.16

    # Conversion parameters to estimate tree cover from  C3 GPP
    gpp_to_tc_a = 15.60
    gpp_to_tc_b = 1.41
    gpp_to_tc_c = -7.72
    c3_forest_closure_gpp = 2.8


@dataclass(frozen=True)
class TModelTraits(ParamClass):
    r"""Trait data settings for a TTree instance.

    This data class provides the value of the key traits used in the T model.
    The default values are taken from Table 1 of :cite:`Li:2014bc`. Note that
    the foliage maintenance respiration fraction is not named in the T Model
    description, but has been included as a modifiable trait in this
    implementation. The traits are shown below with mathematical notation,
    default value and units shown in brackets:

    * `a_hd`: Initial slope of height–diameter relationship (:math:`a`, 116.0, -)
    * `ca_ratio`: Initial ratio of crown area to stem cross-sectional area
        (:math:`c`, 390.43, -)
    * `h_max`: Maximum tree height (:math:`H_m`, 25.33, m)
    * `rho_s`: Sapwood density (:math:`\rho_s`, 200.0, kg Cm−3)
    * `lai`: Leaf area index within the crown (:math:`L`, 1.8, -)
    * `sla`: Specific leaf area (:math:`\sigma`, 14.0, m2 kg−1 C)
    * `tau_f` : Foliage turnover time (:math:`\tau_f`, 4.0, years)
    * `tau_r` : Fine-root turnover time (:math:`\tau_r`, 1.04, years)
    * `par_ext`: PAR extinction coefficient (:math:`k`, 0.5, -)
    * `yld`: Yield_factor (:math:`y`, 0.17, -)
    * `zeta`: Ratio of fine-root mass to foliage area (:math:`\zeta`, 0.17, kg C m−2)
    * `resp_r`: Fine-root specific respiration rate (:math:`r_r`, 0.913, year−1)
    * `resp_s`: Sapwood-specific respiration rate (:math:`r_s`, 0.044, year−1)
    * `resp_f`: Foliage maintenance respiration fraction (:math:`r_f`,  0.1, -)
    """

    a_hd: float = 116.0  # a, Initial slope of height–diameter relationship (-)
    ca_ratio: float = (
        390.43  # c, Initial ratio of crown area to stem cross-sectional area (-)
    )
    h_max: float = 25.33  # H_m, Maximum tree height (m)
    rho_s: float = 200.0  # rho_s, Sapwood density (kgCm−3)
    lai: float = 1.8  # L, Leaf area index within the crown (–)
    sla: float = 14.0  # sigma, Specific leaf area (m2 kg−1C)
    tau_f: float = 4.0  # tau_f, Foliage turnover time (years)
    tau_r: float = 1.04  # tau_r, Fine-root turnover time (years)
    par_ext: float = 0.5  # k, PAR extinction coefficient (–)
    yld: float = 0.17  # y, Yield_factor (-)
    zeta: float = 0.17  # zeta, Ratio of fine-root mass to foliage area (kgCm−2)
    resp_r: float = 0.913  # r_r, Fine-root specific respiration rate (year−1)
    resp_s: float = 0.044  # r_s, Sapwood-specific respiration rate (year−1)
    resp_f: float = 0.1  # --- , Foliage maintenance respiration fraction (-)

    # TODO: include range + se, or make this another class TraitDistrib
    #       that can yield a Traits instance drawing from that distribution


@dataclass(frozen=True)
class HygroParams(ParamClass):
    r"""Parameters for hygrometric functions.

    This data class provides parameters used :mod:`~pyrealm.utilities`, which
    includes hygrometric conversions

    * `mwr`: The ratio molecular weight of water vapour to dry air
        (:math:`MW_r`, 0.622, -)
    * `magnus_params`: A three tuple of coefficients for the Magnus equation for
      the calculation of saturated vapour pressure.
    * `magnus_option`: Selects one of a set of published coefficients for the
        Magnus equation.
    """

    magnus_coef: NDArray[np.float32] = np.array((611.2, 17.62, 243.12))
    mwr: float = 0.622
    magnus_option: Optional[str] = None

    def __post_init__(self) -> None:
        """Populate parameters from init settings.

        This checks the init inputs and populates magnus_coef from the presets
        if no magnus_coef is specified.

        Returns:
            None
        """
        alts = dict(
            Allen1998=np.array((610.8, 17.27, 237.3)),
            Alduchov1996=np.array((610.94, 17.625, 243.04)),
            Sonntag1990=np.array((611.2, 17.62, 243.12)),
        )

        if self.magnus_option is not None:
            if self.magnus_option not in alts:
                raise (
                    RuntimeError(f"magnus_option must be one of {list(alts.keys())}")
                )
            else:
                object.__setattr__(self, "magnus_coef", alts[self.magnus_option])
        elif self.magnus_coef is not None and len(self.magnus_coef) != 3:
            raise TypeError("magnus_coef must be a tuple of 3 numbers")
        else:
            object.__setattr__(self, "magnus_option", None)
