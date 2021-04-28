from dataclasses import dataclass, asdict
from dacite import from_dict
import enforce_typing
from typing import Tuple
from numbers import Number


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

        return asdict(self)

    def to_json(self, filename):

        with open(filename, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

        return None

    @classmethod
    def from_dict(cls, data):

        return from_dict(cls, data)

    @classmethod
    def from_json(cls, filename):

        with open(filename, 'r') as infile:
            json_data = json.load(infile)

        return cls.from_dict(json_data)


# P Model parameter dataclasses - only the

@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamConstant(ParamClass):
    R: Number = 8.3145
    co: Number = 209476.0
    c_molmass: Number = 12.0107
    Po: Number = 101325.0
    To: Number = 25.0
    L: Number = 0.0065
    G: Number = 9.80665
    Ma: Number = 0.028963
    CtoK: Number = 273.15


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamFisherDial(ParamClass):
    lambda_: Tuple[Number, ...] = (1788.316, 21.55053, -0.4695911,
                                  0.003096363, -7.341182e-06)
    Po: Tuple[Number, ...] = (5918.499, 58.05267, -1.1253317,
                             0.0066123869, -1.4661625e-05)
    Vinf: Tuple[Number, ...] = (
        0.6980547, -0.0007435626, 3.704258e-05,
        -6.315724e-07, 9.829576e-09, -1.197269e-10,
        1.005461e-12, -5.437898e-15, 1.69946e-17,
        -2.295063e-20)


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamHuber(ParamClass):
    tk_ast: Number = 647.096
    rho_ast: Number = 322.0
    mu_ast: Number =1e-06
    H_i: Tuple[Number,...] = (1.67752, 2.20462, 0.6366564, -0.241605)
    H_ij: Tuple[Tuple[Number, ...], ...] = (
        (0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0),
         (0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573),
         (-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0),
         (0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0),
         (-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0),
         (0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0),
         (0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264))


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamHeskel(ParamClass):
    b: Number = 0.1012
    c: Number = 0.0005


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamKattgeKnorr(ParamClass):
    a_ent: Number = 668.39
    b_ent: Number = -1.07
    Ha: Number = 71513
    Hd: Number = 200000


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamKphio(ParamClass):
    C4: Tuple[Number,...] = (-0.064, 0.03, -0.000464)
    C3: Tuple[Number, ...] = (0.352, 0.022, -0.00034)


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamBernacchi(ParamClass):
    dhac: Number = 79430
    dhao: Number = 36380
    dha: Number = 37830
    kc25: Number = 39.97
    ko25: Number = 27480
    gs25_0: Number = 4.332


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamSoilmstress(ParamClass):
    theta0: Number = 0.0
    thetastar: Number = 0.6
    a: Number = 0.0
    b: Number = 0.685


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamStocker19(ParamClass):
    beta: Number = 146.0


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamWang17(ParamClass):
    c: Number = 0.41


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamSmith19(ParamClass):
    theta: Number = 0.85
    c_cost: Number = 0.05336251


@enforce_typing.enforce_types
@dataclass(frozen=True)
class ParamAtkin(ParamClass):
    rd_to_vcmax: Number = 0.015


@enforce_typing.enforce_types
@dataclass(frozen=True)
class Param(ParamClass):

    k: ParamConstant = ParamConstant()
    fisher_dial: ParamFisherDial = ParamFisherDial()
    huber: ParamHuber = ParamHuber()
    heskel: ParamHeskel = ParamHeskel()
    kattge_knorr: ParamKattgeKnorr = ParamKattgeKnorr()
    kphio: ParamKphio = ParamKphio()
    bernacchi: ParamBernacchi = ParamBernacchi()
    soilmstress: ParamSoilmstress = ParamSoilmstress()
    stocker19: ParamStocker19 = ParamStocker19()
    wang17: ParamWang17 = ParamWang17()
    smith19: ParamSmith19 = ParamSmith19()
    atkin: ParamAtkin = ParamAtkin()


# T model param class

@enforce_typing.enforce_types
@dataclass(frozen=True)
class Traits(ParamClass):
    """Trait data settings for a TTree instance

    This data class provides the value of the key traits used in the T model.
    The default values are taken from Table 1 of :cite:`Li:2014bc`. The traits
    are:

    * Initial slope of height–diameter relationship (-, `a_hd`,  $a$, 116.0)
    * Initial ratio of crown area to stem cross-sectional area (-, `ca_ratio`,  $c$, 390.43)
    * Maximum tree height (m, `h_max`,  $H_m$, 25.33)
    * Sapwood density (kg Cm−3, `rho_s`,  $\rho_s$, 200.0)
    * Leaf area index within the crown (–, `lai`,  $L$, 1.8)
    * Specific leaf area (m2 kg−1C, `sla`,  $\sigma$, 14.0)
    * Foliage turnover time (years, `tau_f`,  $\tau_f$, 4.0)
    * Fine-root turnover time (years, `tau_r`,  $\tau_r$, 1.04)
    * PAR extinction coefficient (–, `par_ext`,  $k$, 0.5)
    * Yield_factor (-, `yld`,  $y$, 0.17)
    * Ratio of fine-root mass to foliage area (kgCm−2, `zeta`,  $\zeta$, 0.17)
    * Fine-root specific respiration rate (year−1, `resp_r`,  $r_r$, 0.913)
    * Sapwood-specific respiration rate (year−1, `resp_s`,  $r_s$, 0.044)
    * Foliage maintenance respiration fraction (-, `resp_f`,  $r_f$  0.1)
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
