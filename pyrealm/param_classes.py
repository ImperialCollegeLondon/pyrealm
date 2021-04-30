from dataclasses import dataclass, asdict
from typing import Tuple
from numbers import Number
import json
import enforce_typing
from dacite import from_dict


class ParamClass:
    """Base class for model parameter classes

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

    def to_dict(self):
        """Returns a dictionary of the attributes and values used in an instance
        of a parameter class object.

        Returns:
            A dictionary
        """
        return asdict(self)

    def to_json(self, filename):
        """Writes the attributes and values used in an instance of a parameter
        class object to a JSON file.

        Args:
            filename: A path to the JSON file to be created.

        Returns:
            None
        """
        with open(filename, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

        return None

    @classmethod
    def from_dict(cls, data):
        """Generates a parameter class object using the data provided in a
        dictionary to override default values

        Args:
            data: A dictionary, keyed by parameter class attribute names,
                  providing values to use in a new instance.

        Returns:
            An instance of the parameter class.
        """
        return from_dict(cls, data)

    @classmethod
    def from_json(cls, filename):
        """Generates a parameter class object using the data provided in a
        JSON file.

        Args:
            filename: The path to a JSON formatted file containing a set
                     of values keyed by parameter class attribute names
                     to use in a new instance.

        Returns:
            An instance of the parameter class.
        """
        with open(filename, 'r') as infile:
            json_data = json.load(infile)

        return cls.from_dict(json_data)


# P Model param class

@enforce_typing.enforce_types
@dataclass(frozen=True)
class PModelParams(ParamClass):

    r"""Model parameters for the P Model

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
    * `k_Ma`: Molecular weight of dry air (Tsilingiris, 2008)  (:math:`M_a`, 0.028963, kg/mol)
    * `l_CtoK`: Conversion from °C to K   (:math:`CtoK` , 273.15, -)

    **Density of water**, values taken from Table 5 of :cite:`Fisher:1975tm`.

    * `fisher_dial_lambda`: [1788.316, 21.55053, -0.4695911, 3.096363e-3, -7.341182e-6]
    * `fisher_dial_Po`: [5918.499, 58.05267, -1.1253317, 6.613869e-3, -1.4661625e-5]
    * `fisher_dial_Vinf`: [0.6980547, -7.435626e-4, 3.704258e-5, -6.315724e-7,
      9.829576e-9, -1.197269e-10, 1.005461e-12, -5.437898e-15,
      1.69946e-17, -2.295063e-20]

    **Viscosity of water**, values taken from :cite:`Huber:2009fy`

    * `huber_tk_ast`: Reference temperature (:math:`tk_{ast}`, 647.096, Kelvin)
    * `huber_rho_ast`: Reference density (:math:`\rho_{ast}`, 322.0, kg/m^3)
    * `huber_mu_ast`: Reference pressure (:math:`\mu_{ast}` 1.0e-6, Pa s)
    * `huber_H_i`: Values of H_i (Table 2): (:math:`H_i`, [1.67752, 2.20462, 0.6366564, -0.241605])
    * `huber_H_ij`: Values of H_ij (Table 3): (:math:`H_ij`,
      [[0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
      [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
      [-0.281378, -0.906851, -0.772479, -0.489837, -0.257040, 0.0],
      [0.161913,  0.257399, 0.0, 0.0, 0.0, 0.0],
      [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
      [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264]])

    **Temperature scaling of dark respiration**, values taken from :cite:`Heskel:2016fg`

    * `heskel_b`: Linear factor in scaling (:math:`b`, 0.1012)
    * `heskel_c`: Quadratic factor in scaling (:math:`c`, 0.0005)

    **Temperature and entropy of VCMax**, values taken from Table 3 of :cite:`Kattge:2007db`

    * `kattge_knorr_a_ent`: Offset of entropy vs. temperature relationship (:math:`a_{ent}`, 668.39, J/mol/K)
    * `kattge_knorr_b_ent`: Slope of entropy vs. temperature relationship (:math:`b_{ent}`, -1.07, J/mol/K^2)
    * `kattge_knorr_Ha`: Activation energy (:math:`H_a`, 71513, J/mol)
    * `kattge_knorr_Hd`: Deactivation energy (:math:`H_d`, 200000, J/mol)

    **Scaling of Kphio with temperature**, parameters of quadratic functions

    * `kphio_C4`: Scaling of Kphio in C4 plants, unpublished estimates
      from Shirley (Cai, Wenjia <w.cai17@imperial.ac.uk>) ([-0.064,  0.03, -0.000464])
    * `kphio_C3`: Scaling of Kphio in C3 plants, taken from Table 2 of
      :cite:`Bernacchi:2003dc` ([0.352, 0.022, -3.4e-4])

    **Temperature responses of photosynthetic enzymes**, values taken from Table 1
    of :cite:`Bernacchi:2003dc`

    * dhac: 79430  # (J/mol) Activation energy (Kc)
    * dhao: 36380  # (J/mol) Activation energy (Ko)
    * dha: 37830  # (J/mol) Activation energy (gamma*)
    * # k25 parameters are not dependent on atmospheric pressure, value converted
    * # to Pa by T. Davis assuming elevation of 227.076 m.a.s.l. = 98716.403 Pa
    * kc25: 39.97  # Reported as 404.9 µmol mol-1, converted as 0.0004049 x 98716.403 = 39.97 Pa
    * ko25: 27480  # Reported as 278.4 mmol mol-1, converted as 0.2784 x 98716.403 = 27480 Pa
    * # Reported as 42.75 µmol mol-1,  converted using 42.75 p_0 = 4.332 Pa`
    * gs25_0: 4.332  # Pa

    **Soil moisture stress**, parameterisation from :cite:`Stocker:2020dh`

    * `soilmstress_theta0`: 0.0
    * `soilmstress_thetastar`: 0.6
    * `soilmstress_a`: 0.0
    * `soilmstress_b`: 0.685

    **Unit cost ratio**, value taken from :cite:`Stocker:2020dh`

    * `stocker19_beta`: Unit cost ratio. (:math:`\beta`, 146.0)

    **Electron transport capacity maintenance cost**, value taken from :cite:`Wang:2017go`

    *  `wang_c`: unit carbon cost for the maintenance of electron transport capacity (:math:`c`, 0.41, )

    **Calculation of omega**, scaling factor in J max limitation method of :cite:`Smith:2019dv`:

    * `smith19_theta`: (0.85)
    * `smith19_c_cost`: (0.05336251)

    **Dark respiration**, values taken from :cite:`Atkin:2015hk` for C3 herbaceous plants:

    * `atkin_rd_to_vcmax`:  Ratio of Rdark to Vcmax25 (0.015)
    """

    # Constants
    k_R: Number = 8.3145
    k_co: Number = 209476.0
    k_c_molmass: Number = 12.0107
    k_Po: Number = 101325.0
    k_To: Number = 25.0
    k_L: Number = 0.0065
    k_G: Number = 9.80665
    k_Ma: Number = 0.028963
    k_CtoK: Number = 273.15
    # Fisher Dial
    fisher_dial_lambda: Tuple[Number, ...] = (
        1788.316, 21.55053, -0.4695911, 0.003096363, -7.341182e-06)
    fisher_dial_Po: Tuple[Number, ...] = (
        5918.499, 58.05267, -1.1253317, 0.0066123869, -1.4661625e-05)
    fisher_dial_Vinf: Tuple[Number, ...] = (
        0.6980547, -0.0007435626, 3.704258e-05, -6.315724e-07, 9.829576e-09,
        -1.197269e-10, 1.005461e-12, -5.437898e-15, 1.69946e-17, -2.295063e-20)
    # Huber
    huber_tk_ast: Number = 647.096
    huber_rho_ast: Number = 322.0
    huber_mu_ast: Number =1e-06
    huber_H_i: Tuple[Number,...] = (1.67752, 2.20462, 0.6366564, -0.241605)
    huber_H_ij: Tuple[Tuple[Number, ...], ...] = (
        (0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0),
        (0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573),
        (-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0),
        (0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0),
        (-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0),
        (0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264))
    # Heskel
    heskel_b: Number = 0.1012
    heskel_c: Number = 0.0005
    # KattgeKnorr
    kattge_knorr_a_ent: Number = 668.39
    kattge_knorr_b_ent: Number = -1.07
    kattge_knorr_Ha: Number = 71513
    kattge_knorr_Hd: Number = 200000
    # Kphio:
    # - note that kphio_C4 has been updated to account for an unintended double
    #   8 fold downscaling to account for the fraction of light reaching PS2.
    #   from original values of [-0.008, 0.00375, -0.58e-4]
    kphio_C4: Tuple[Number, ...] = (-0.064, 0.03, -0.000464)
    kphio_C3: Tuple[Number, ...] = (0.352, 0.022, -0.00034)
    # Bernachhi
    bernacchi_dhac: Number = 79430
    bernacchi_dhao: Number = 36380
    bernacchi_dha: Number = 37830
    bernacchi_kc25: Number = 39.97
    bernacchi_ko25: Number = 27480
    bernacchi_gs25_0: Number = 4.332
    # Soilmstress
    soilmstress_theta0: Number = 0.0
    soilmstress_thetastar: Number = 0.6
    soilmstress_a: Number = 0.0
    soilmstress_b: Number = 0.685
    # Stocker19
    stocker19_beta: Number = 146.0
    # Wang17
    wang17_c: Number = 0.41
    # Smith19
    smith19_theta: Number = 0.85
    smith19_c_cost: Number = 0.05336251
    # Atkin
    atkin_rd_to_vcmax: Number = 0.015


# T model param class

@enforce_typing.enforce_types
@dataclass(frozen=True)
class TModelTraits(ParamClass):
    r"""Trait data settings for a TTree instance

    This data class provides the value of the key traits used in the T model.
    The default values are taken from Table 1 of :cite:`Li:2014bc`. Note that
    the foliage maintenance respiration fraction is not named in the T Model
    description, but has been included as a modifiable trait in this
    implementation. The traits are shown below with mathematical notation,
    default value and units shown in brackets:

    * `a_hd`: Initial slope of height–diameter relationship (:math:`a`, 116.0, -)
    * `ca_ratio`: Initial ratio of crown area to stem cross-sectional area (:math:`c`, 390.43, -)
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

    a_hd: Number = 116.0        # a, Initial slope of height–diameter relationship (-)
    ca_ratio: Number = 390.43   # c, Initial ratio of crown area to stem cross-sectional area (-)
    h_max: Number = 25.33       # H_m, Maximum tree height (m)
    rho_s: Number = 200.0       # rho_s, Sapwood density (kgCm−3)
    lai: Number = 1.8           # L, Leaf area index within the crown (–)
    sla: Number = 14.0          # sigma, Specific leaf area (m2 kg−1C)
    tau_f: Number = 4.0         # tau_f, Foliage turnover time (years)
    tau_r: Number = 1.04        # tau_r, Fine-root turnover time (years)
    par_ext: Number = 0.5       # k, PAR extinction coefficient (–)
    yld: Number = 0.17          # y, Yield_factor (-)
    zeta: Number = 0.17         # zeta, Ratio of fine-root mass to foliage area (kgCm−2)
    resp_r: Number = 0.913      # r_r, Fine-root specific respiration rate (year−1)
    resp_s: Number = 0.044      # r_s, Sapwood-specific respiration rate (year−1)
    resp_f: Number = 0.1        # --- , Foliage maintenance respiration fraction (-)

    # TODO include range + se, or make this another class TraitDistrib
    #      that can yield a Traits instance drawing from that distribution
