"""The ``t_model`` module provides the basic scaling relationships of the T Model
:cite:`Li:2014bc`. This provides scaling relationships using the plant functional type
traits defined in the :mod:`~pyrealm.demography.flora` module and the diameter at breast
height of individual stems to define the stem geometry, masses, respiration and hence
calculate stem growth given net primary productivity. Note that
:attr:`~pyrealm.demography.tmodel.StemAllometry.stem_height` denotes the total tree
height, as used interchangeable in :cite:`Li:2014bc`, rather than just the height of the
trunk below the canopy.
"""  # noqa: D205

from dataclasses import InitVar, dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.demography.core import (
    CohortMethods,
    PandasExporter,
    _enforce_2D,
    _validate_demography_array_arguments,
)
from pyrealm.demography.flora import Flora, StemTraits


def _enforce_positive_sizes(size_args: dict[str, NDArray], function_name: str) -> None:
    """Simple function to trap allometry inputs that are not strictly positive."""

    failing = [vname for vname, values in size_args.items() if np.any(values <= 0)]

    if failing:
        raise ValueError(
            f"Allometry values in {function_name} not strictly "
            f"positive: {', '.join(failing)}"
        )


def calculate_heights(
    h_max: NDArray[np.float64],
    a_hd: NDArray[np.float64],
    dbh: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate tree height under the T Model.

    The height of trees (:math:`H`) are calculated from individual diameters at breast
    height (:math:`D`), along with the maximum height (:math:`H_{m}`) and initial slope
    of the height/diameter relationship (:math:`a`) of the plant functional types
    :cite:p:`{Equation 4, }Li:2014bc`:

    .. math::

         H = H_{m}  \left(1 - \exp(-a \cdot D / H_{m})\right)

    Args:
        h_max: Maximum height of the PFT
        a_hd: Initial slope of the height/diameter relationship of the PFT
        dbh: Diameter at breast height of individuals
        validate: Boolean flag to suppress argument validation
    """

    if validate:
        size_args = {"dbh": dbh}
        _validate_demography_array_arguments(
            trait_args={"h_max": h_max, "a_hd": a_hd}, size_args=size_args
        )
        _enforce_positive_sizes(size_args=size_args, function_name="calculate_heights")

    return _enforce_2D(h_max * (1 - np.exp(-a_hd * dbh / h_max)))


def calculate_dbh_from_height(
    h_max: NDArray[np.float64],
    a_hd: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate diameter at breast height from stem height under the T Model.

    This function inverts the normal calculation of stem height (:math:`H`) from
    diameter at breast height (DBH, :math:`D`) in the T Model (see
    :meth:`~pyrealm.demography.tmodel.calculate_heights`). This is a helper
    function to allow users to convert known stem heights for a plant functional type,
    with maximum height (:math:`H_{m}`) and initial slope of the height/diameter
    relationship (:math:`a`) into the expected DBH values.

    .. math::

         D = \frac{H \left( \log \left(\frac{H}{H_{m}-H}\right)\right)}{a}

    Warning:
        Where the stem height is greater than the maximum height for a PFT, then
        DBH is undefined and the return array will contain `np.nan`. Where the
        stem height equals the maximum height, the model predicts an infinite stem
        diameter: the `h_max` parameter is the asymptotic maximum stem height of an
        exponential function. Similarly, heights very close to the maximum height may
        lead to unrealistically large predictions of DBH.

    Args:
        h_max: Maximum height of the PFT
        a_hd: Initial slope of the height/diameter relationship of the PFT
        stem_height: Stem height of individuals
        validate: Boolean flag to suppress argument validation
    """

    if validate:
        size_args = {"stem_height": stem_height}
        _validate_demography_array_arguments(
            trait_args={"h_max": h_max, "a_hd": a_hd}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_dbh_from_height"
        )

    # The equation here blows up in a couple of ways:
    # - H > h_max leads to negative logs which generates np.nan with an invalid value
    #   warning. The np.nan here is what we want to happen, so the warning needs
    #   suppressing.
    # - H = h_max generates a divide by zero which returns inf with a warning. Here the
    #   answer should be h_max so that needs trapping.

    with np.errstate(divide="ignore", invalid="ignore"):
        return _enforce_2D((h_max * np.log(h_max / (h_max - stem_height))) / a_hd)


def calculate_crown_areas(
    ca_ratio: NDArray[np.float64],
    a_hd: NDArray[np.float64],
    dbh: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate tree crown area under the T Model.

    The tree crown area (:math:`A_{c}`) is calculated from individual diameters at
    breast height (:math:`D`) and stem height (:math:`H`), along with the crown area
    ratio (:math:`c`) and the initial slope of the height/diameter relationship
    (:math:`a`) of the plant functional type :cite:p:`{Equation 8, }Li:2014bc`:

    .. math::

        A_{c} =\frac{\pi c}{4 a} D H


    Args:
        ca_ratio: Crown area ratio of the PFT
        a_hd: Initial slope of the height/diameter relationship of the PFT
        dbh: Diameter at breast height of individuals
        stem_height: Stem height of individuals
        validate: Boolean flag to suppress argument validation
    """

    if validate:
        size_args = {"dbh": dbh, "stem_height": stem_height}

        _validate_demography_array_arguments(
            trait_args={"ca_ratio": ca_ratio, "a_hd": a_hd}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_crown_areas"
        )

    return _enforce_2D(((np.pi * ca_ratio) / (4 * a_hd)) * dbh * stem_height)


def calculate_crown_fractions(
    a_hd: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    dbh: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate tree crown fraction under the T Model.

    The crown fraction (:math:`f_{c}`) is calculated from individual diameters at breast
    height (:math:`D` for :math:`D > 0`) and stem height (:math:`H`), along with the
    initial slope of the height / diameter relationship (:math:`a`) of the plant
    functional type :cite:p:`{Equation 11, }Li:2014bc`:

    .. math::

        f_{c} =\frac{H}{a D}

    Args:
        a_hd: Initial slope of the height/diameter relationship of the PFT
        stem_height: Stem height of individuals
        dbh: Diameter at breast height of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"dbh": dbh, "stem_height": stem_height}
        _validate_demography_array_arguments(
            trait_args={"a_hd": a_hd}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_crown_fractions"
        )

    # Calculate crown fraction
    return _enforce_2D(stem_height / (a_hd * dbh))


def calculate_stem_masses(
    rho_s: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    dbh: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate stem mass under the T Model.

    The stem mass (:math:`W_{s}`) is calculated from individual diameters at breast
    height (:math:`D`) and stem height (:math:`H`), along with the wood density
    (:math:`\rho_s`) of the plant functional type :cite:p:`{Equation 6, }Li:2014bc`:

    .. math::

        W_s = (\pi / 8) \rho_s D^2 H

    Args:
        rho_s: Wood density of the PFT
        stem_height: Stem height of individuals
        dbh: Diameter at breast height of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"dbh": dbh, "stem_height": stem_height}
        _validate_demography_array_arguments(
            trait_args={"rho_s": rho_s}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_stem_masses"
        )

    return _enforce_2D((np.pi / 8) * rho_s * (dbh**2) * stem_height)


def calculate_foliage_masses(
    sla: NDArray[np.float64],
    lai: NDArray[np.float64],
    crown_area: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate foliage mass under the T Model.

    The foliage mass (:math:`W_{f}`) is calculated from the crown area (:math:`A_{c}`),
    along with the specific leaf area (:math:`\sigma`) and leaf area index (:math:`L`)
    of the plant functional type :cite:p:`Li:2014bc`.

    .. math::

        W_f = (1 / \sigma) A_c L

    Args:
        sla: Specific leaf area of the PFT
        lai: Leaf area index of the PFT
        crown_area: Crown area of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"crown_area": crown_area}
        _validate_demography_array_arguments(
            trait_args={"sla": sla, "lai": lai}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_foliage_masses"
        )

    return _enforce_2D(crown_area * lai * (1 / sla))


def calculate_sapwood_masses(
    rho_s: NDArray[np.float64],
    ca_ratio: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    crown_area: NDArray[np.float64],
    crown_fraction: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate sapwood mass under the T Model.

    The sapwood mass (:math:`W_{\cdot s}`) is calculated from the individual crown area
    (:math:`A_{c}`), stem height (:math:`H`) and canopy fraction (:math:`f_{c}`) along
    with the wood density (:math:`\rho_s`) and crown area ratio (:math:`c`) of the plant
    functional type, following Equation 14 of :cite:`Li:2014bc`. The function is
    undefined for negative or zero heights.

    .. math::

        W_{\cdot s} = \frac{A_c \rho_s H (1 - f_c / 2)}{c}

    Args:
        rho_s: Wood density of the PFT
        ca_ratio: Crown area ratio of the PFT
        stem_height: Stem height of individuals
        crown_area: Crown area of individuals
        crown_fraction: Crown fraction of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {
            "stem_height": stem_height,
            "crown_area": crown_area,
            "crown_fraction": crown_fraction,
        }
        _validate_demography_array_arguments(
            trait_args={"rho_s": rho_s, "ca_ratio": ca_ratio}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_sapwood_masses"
        )

    return _enforce_2D(
        crown_area * rho_s * stem_height * (1 - crown_fraction / 2) / ca_ratio,
    )


def calculate_crown_z_max(
    z_max_prop: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate height of maximum crown radius.

    The height of the maximum crown radius (:math:`z_m`) is derived from the crown
    shape parameters (:math:`m,n`) and the resulting fixed proportion (:math:`p_{zm}`)
    for plant functional types. These shape parameters are defined as part of the
    extension of the T Model presented by :cite:t:`joshi:2022a`.

    The value :math:`z_m` is the height above ground where the largest crown radius is
    found, given the proportion and the estimated stem height (:math:`H`) of
    individuals.

    .. math::

        z_m = p_{zm} H

    Args:
        z_max_prop: Crown shape parameter of the PFT
        stem_height: Stem height of individuals
        validate: Boolean flag to suppress argument validation
    """

    if validate:
        size_args = {"stem_height": stem_height}
        _validate_demography_array_arguments(
            trait_args={"z_max_prop": z_max_prop}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_crown_z_max"
        )

    return _enforce_2D(stem_height * z_max_prop)


def calculate_crown_r0(
    q_m: NDArray[np.float64],
    crown_area: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate scaling factor for width of maximum crown radius.

    This scaling factor (:math:`r_0`) is derived from the crown shape parameters
    (:math:`m,n,q_m`) for plant functional types and the estimated crown area
    (:math:`A_c`) of individuals. The shape parameters are defined as part of the
    extension of the T Model presented by :cite:t:`joshi:2022a` and :math:`r_0` is used
    to scale the crown area such that the crown area at the  maximum crown radius fits
    the expectations of the T Model.

    .. math::

        r_0 = 1/q_m  \sqrt{A_c / \pi}

    Args:
        q_m: Crown shape parameter of the PFT
        crown_area: Crown area of individuals
        validate: Boolean flag to suppress argument validation

    """

    if validate:
        size_args = {"crown_area": crown_area}
        _validate_demography_array_arguments(
            trait_args={"q_m": q_m}, size_args=size_args
        )
        _enforce_positive_sizes(size_args=size_args, function_name="calculate_crown_r0")

    # Scaling factor to give expected A_c (crown area) at
    # z_m (height of maximum crown radius)
    return _enforce_2D(1 / q_m * np.sqrt(crown_area / np.pi))


def calculate_whole_crown_gpp(
    potential_gpp: NDArray[np.float64],
    crown_area: NDArray[np.float64],
    par_ext: NDArray[np.float64],
    lai: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate whole crown gross primary productivity.

    This function calculates individual gross primary productivity (GPP) across the
    whole crown, given the individual potential GPP per metre squared (:math:`P_0`, kg C
    m-2) and crown area (:math:`A_c`, m2), along with the leaf area index (:math:`L`)
    and the extinction coefficient (:math:`k`) of the plant functional type
    :cite:p:`{Equation 12, }Li:2014bc`.

    .. math::

        P = P_0 A_c (1 - e^{-kL})

    Args:
        lai: The leaf area index
        par_ext: The extinction coefficient
        potential_gpp: Potential GPP per metre squared
        crown_area: The crown area in metres squared
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"potential_gpp": potential_gpp, "crown_area": crown_area}
        _validate_demography_array_arguments(
            trait_args={"lai": lai, "par_ext": par_ext}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_whole_crown_gpp"
        )

    return _enforce_2D(potential_gpp * crown_area * (1 - np.exp(-(par_ext * lai))))


def calculate_sapwood_respiration(
    resp_s: NDArray[np.float64],
    sapwood_mass: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate sapwood respiration.

    Calculates the total sapwood respiration (:math:`R_{\cdot s}`) given the individual
    sapwood mass (:math:`W_{\cdot s}`) and the sapwood respiration rate of the plant
    functional type (:math:`r_{s}`) :cite:p:`{see Equation 13, }Li:2014bc`.

    .. math::
         R_{\cdot s} = W_{\cdot s} \, r_s

    Args:
        resp_s: The sapwood respiration rate
        sapwood_mass: The individual sapwood mass
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"sapwood_mass": sapwood_mass}
        _validate_demography_array_arguments(
            trait_args={"resp_s": resp_s}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_sapwood_respiration"
        )

    return _enforce_2D(sapwood_mass * resp_s)


def calculate_foliar_respiration(
    resp_f: NDArray[np.float64],
    whole_crown_gpp: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate foliar respiration.

    Calculates the total foliar respiration (:math:`R_{f}`) given the individual crown
    GPP (:math:`P`) and the foliar respiration rate of the plant functional type
    (:math:`r_{f}`). :cite:t:`Li:2014bc` remove foliar respiration as a constant
    proportion of potential GPP before calculating GPP for the crown, but ``pyrealm``
    treats this proportion as part of the definition of plant functional types.

    .. math::
         R_{f} = P \, r_f

    Args:
        resp_f: The foliar respiration rate
        whole_crown_gpp: The individual whole crown GPP.
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"whole_crown_gpp": whole_crown_gpp}
        _validate_demography_array_arguments(
            trait_args={"resp_f": resp_f}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_foliar_respiration"
        )

    return _enforce_2D(whole_crown_gpp * resp_f)


def calculate_gpp_topslice(
    gpp_topslice: NDArray[np.float64],
    whole_crown_gpp: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate gpp topslice.

    Calculates a fixed proportion of the total GPP for the crown that is removed before
    further GPP allocation. This is intended as a helper variable for T Model users to
    simulate processes not included in the T Model such as root exudation or active
    nutrient servicing for mycorriza fungi.

    .. NOTE::

        This is a naive calculation method that is not part of the T model. If values
        for GPP topslice are zero it will have no impact on the T Model calculations.

    Args:
        gpp_topslice: The portion of GPP to remove before allocation.
        whole_crown_gpp: The individual whole crown GPP.
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"whole_crown_gpp": whole_crown_gpp}
        _validate_demography_array_arguments(
            trait_args={"gpp_topslice": gpp_topslice}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_gpp_topslice"
        )

    return _enforce_2D(whole_crown_gpp * gpp_topslice)


def calculate_reproductive_tissue_respiration(
    resp_rt: NDArray[np.float64],
    reproductive_tissue_mass: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate reproductive tissue respiration.

    Calculates the total reproductive tissue respiration (:math:`R_{rt}`) given the
    reproductive tissue mass (:math:`M_rt`) and the reproductive tissue respiration rate
    of the plant functional type (:math:`r_{rt}`).

    NOTE: This function is not part of the original T Model, but is included here to
    allow for the calculation of reproductive tissue respiration in the same way as
    sapwood respiration.

    .. math::
         R_{rt} = M_rt \, r_rt

    Args:
        resp_rt: The reproductive tissue respiration rate
        reproductive_tissue_mass: The stem reproductive tissue mass.
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"reproductive_tissue_mass": reproductive_tissue_mass}
        _validate_demography_array_arguments(
            trait_args={"resp_rt": resp_rt}, size_args=size_args
        )
        if np.any(reproductive_tissue_mass < 0):
            raise ValueError("The reproductive tissue mass cannot be negative.")

    return _enforce_2D(reproductive_tissue_mass * resp_rt)


def calculate_fine_root_respiration(
    zeta: NDArray[np.float64],
    sla: NDArray[np.float64],
    resp_r: NDArray[np.float64],
    foliage_mass: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate fine root respiration.

    Calculates the total fine root respiration (:math:`R_{r}`) given the individual
    foliage mass (:math:`W_f`), along with the fine root respiration rate (:math:`r_r`),
    the ratio of fine root mass to foliage area (:math:`\zeta`) and the specific leaf
    area (:math:`\sigma`) :cite:p:`{see Equation 13, }Li:2014bc`

    .. math::
         R_{r} = \zeta \sigma W_f r_r

    Args:
        zeta: The ratio of fine root mass to foliage area of the PFT.
        sla: The specific leaf area of the PFT.
        resp_r: The respiration rate of fine roots of the PFT.
        foliage_mass: The individual foliage mass.
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"foliage_mass": foliage_mass}
        _validate_demography_array_arguments(
            trait_args={"zeta": zeta, "sla": sla, "resp_r": resp_r},
            size_args=size_args,
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_fine_root_respiration"
        )

    return _enforce_2D(zeta * sla * foliage_mass * resp_r)


def calculate_net_primary_productivity(
    yld: NDArray[np.float64],
    whole_crown_gpp: NDArray[np.float64],
    foliar_respiration: NDArray[np.float64],
    fine_root_respiration: NDArray[np.float64],
    sapwood_respiration: NDArray[np.float64],
    reproductive_tissue_respiration: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate net primary productivity.

    The net primary productivity (NPP, :math:`P_{net}`) is calculated as a plant
    functional type specific yield proportion (:math:`y`) of the total GPP (:math:`P`)
    for the individual minus respiration (:math:`R_m`), as the sum of the respiration
    costs for foliage  (:math:`R_f`), fine roots  (:math:`R_r`), sapwood
    (:math:`R_s`), and reproductive tissue (:math:`R_{rt}`).

    .. math::
        P_{net} = y (P - R_m) = y (P - W_{\cdot s} r_s - \zeta \sigma W_f r_r - W_f r_f
        - P r_{rt})

    Note that this differs from Equation 13 of :cite:t:`Li:2014bc`, which does not
    include a term for foliar respiration or reproductive tissue respiration.
    :cite:t:`Li:2014bc` remove foliar respiration as a fixed proportion of potential GPP
    as the first step in their calculations. The approach here is equivalent but allows
    the foliar respiration to vary between plant functional types. :cite:t:`Li:2014bc`
    do not include reproductive tissue respiration in their calculations.

    Args:
        yld: The yield proportion.
        whole_crown_gpp: The total GPP for the crown.
        foliar_respiration: The total foliar respiration.
        fine_root_respiration: The total fine root respiration
        sapwood_respiration: The total sapwood respiration.
        reproductive_tissue_respiration: The total reproductive tissue respiration.
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {
            "whole_crown_gpp": whole_crown_gpp,
            "foliar_respiration": foliar_respiration,
            "fine_root_respiration": fine_root_respiration,
            "sapwood_respiration": sapwood_respiration,
            "reproductive_tissue_respiration": reproductive_tissue_respiration,
        }
        _validate_demography_array_arguments(
            trait_args={"yld": yld}, size_args=size_args
        )
        # Allow reproductive tissue terms to be zero
        size_args = {
            k: v
            for k, v in size_args.items()
            if k not in ("reproductive_tissue_respiration")
        }
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_net_primary_productivity"
        )

    return _enforce_2D(
        yld
        * (
            whole_crown_gpp
            - foliar_respiration
            - fine_root_respiration
            - sapwood_respiration
            - reproductive_tissue_respiration
        )
    )


def calculate_foliage_turnover(
    tau_f: NDArray[np.float64],
    foliage_mass: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate turnover costs for foliage.

    This function calculates the costs associated with the turnover of foliage. This is
    calculated from the total foliage mass of individuals (:math:`W_f`), and the
    turnover times of foliage (:math:`\tau_f`) of the plant functional type
    :cite:p:`{see Equation 15, }Li:2014bc`.

    .. math::

        T = W_f \left( \frac{1}{\tau_f} \right)

    Args:
        tau_f: The turnover time of foliage
        foliage_mass: The foliage mass
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"foliage_mass": foliage_mass}
        _validate_demography_array_arguments(
            trait_args={"tau_f": tau_f},
            size_args=size_args,
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_foliage_turnover"
        )

    return _enforce_2D(foliage_mass * (1 / tau_f))


def calculate_fine_root_turnover(
    sla: NDArray[np.float64],
    zeta: NDArray[np.float64],
    tau_r: NDArray[np.float64],
    foliage_mass: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate turnover costs.

    This function calculates the costs associated with the turnover of fine roots. This
    is calculated from the total foliage mass of individuals (:math:`W_f`), along with
    the specific leaf area (:math:`\sigma`) and fine root mass to foliar area ratio
    (:math:`\zeta`) and the turnover time of fine roots (:math:`\tau_r`) of the plant
    functional type :cite:p:`{see Equation 15, }Li:2014bc`.

    .. math::

        T = W_f \left(\frac{ \sigma \zeta}{\tau_f} \right)

    Args:
        sla: The specific leaf area
        zeta: The ratio of fine root mass to foliage area.
        tau_r: The turnover time of fine roots
        foliage_mass: The foliage mass
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {"foliage_mass": foliage_mass}
        _validate_demography_array_arguments(
            trait_args={"sla": sla, "zeta": zeta, "tau_r": tau_r}, size_args=size_args
        )
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_fine_root_turnover"
        )

    return _enforce_2D(foliage_mass * (sla * zeta / tau_r))


def calculate_reproductive_tissue_turnover(
    reproductive_tissue_mass: NDArray[np.float64],
    tau_rt: NDArray[np.float64],
    validate: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate reproductive tissue turnover costs.

    This function calculates the costs associated with the turnover of reproductive
    tissue. This is calculated from the total reproductive tissue mass
    (:math:`m_{rt}`), along with the turnover time of reproductive tissue
    (:math:`\tau_{rt}`).

    .. math::

        T_{rt} = m_{rt} \left( \frac{1}{\tau_{rt}}\right)

    Args:
        reproductive_tissue_mass: The mass of reproductive tissue
        tau_rt: The turnover time of reproductive tissue
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        _validate_demography_array_arguments(
            trait_args={"tau_rt": tau_rt},
            size_args={"reproductive_tissue_mass": reproductive_tissue_mass},
        )
        if np.any(reproductive_tissue_mass < 0):
            raise ValueError("The reproductive tissue mass cannot be negative.")

    return _enforce_2D(reproductive_tissue_mass * (1 / tau_rt))


def calculate_reproductive_tissue_mass(
    foliage_mass: NDArray[np.float64],
    p_foliage_for_reproductive_tissue: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate reproductive tissue mass.

    This function calculates the mass of reproductive tissue (:math:`m_{rt}`) as a fixed
    proportion of the total foliage mass (:math:`W_f`) of individuals.

    .. math::

        m_{rt} = p_{f_{rt}} W_f

    Args:
        foliage_mass: The foliage mass
        p_foliage_for_reproductive_tissue: The proportion of foliage mass that is
            reproductive tissue
        validate: Boolean flag to suppress argument validation
    """

    return _enforce_2D(p_foliage_for_reproductive_tissue * foliage_mass)


def calculate_growth_increments(
    rho_s: NDArray[np.float64],
    a_hd: NDArray[np.float64],
    h_max: NDArray[np.float64],
    lai: NDArray[np.float64],
    ca_ratio: NDArray[np.float64],
    sla: NDArray[np.float64],
    zeta: NDArray[np.float64],
    npp: NDArray[np.float64],
    turnover: NDArray[np.float64],
    reproductive_tissue_turnover: NDArray[np.float64],
    p_foliage_for_reproductive_tissue: NDArray[np.float64],
    dbh: NDArray[np.float64],
    stem_height: NDArray[np.float64],
    validate: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate growth increments.

    Given an estimate of net primary productivity (NPP, :math:`P_{net}`), less
    associated turnover costs (:math:`T`), the remaining productivity can be allocated
    to growth and hence estimate resulting increments :cite:`Li:2014bc` in:
    
    * the stem diameter (:math:`\Delta D`),
    * the stem mass (:math:`\Delta W_s`), and 
    * the foliar mass (:math:`\Delta W_f`). 
        
    The stem diameter increment can be calculated using the available productivity for
    growth and the rates of change in stem (:math:`\textrm{d}W_s / \textrm{d}t`) and
    foliar masses (:math:`\textrm{d}W_f / \textrm{d}t`): 

    .. math::

        \Delta D = \frac{P_{net} - T}{ \textrm{d}W_s / \textrm{d}t  +
             \textrm{d}W_f / \textrm{d}t}

    The rates of change in stem and foliar mass can be calculated as:

    .. math::
      :nowrap:

      \[
        \begin{align*}
            \textrm{d}W_s / \textrm{d}t &= \frac{\pi}{8} \rho_s D
                \left(a D \left(1 - \frac{H}{H_{m}} + 2 H \right) \right) \\

            \textrm{d}W_f / \textrm{d}t &= L \frac{\pi c}{4 a} \left(a D \left( 1 -
                \frac{H}{H_{m}} + H \right) \right) \frac{1}{\sigma + \zeta}
        \end{align*}
      \]

    given the current stem diameter (:math:`D`) and height (:math:`H`) and the following
    plant functional type traits:

    * the specific leaf area (:math:`\sigma`),
    * the leaf area index (:math:`L`),
    * the wood  density of the PFT (:math:`\rho_s`),
    * the maximum height (:math:`H_{m}`),
    * the initial slope of the height/diameter relationship (:math:`a`),
    * the crown area ratio (:math:`c`), and
    * the ratio of fine root mass to leaf area (:math:`\zeta`).

    The value of :math:`\Delta D` is unstable when :math:`D = 0` and hence :math:`H = 0`
    and the rates of change in stem and foliar mass are also zero. If :math:`P_{net} - T
    = 0` then :math:`\Delta D` is undefined, otherwise :math:`\Delta D = \pm \inf`
    depending on whether then turnover costs exceed the available NPP. Under these
    conditions, this function explicitly sets :math:`\Delta D = 0`: **stems with zero
    height cannot grow**.

    The resulting incremental changes in stem mass and foliar mass can then be
    calculated as:

    .. math::
      :nowrap:

      \[
        \begin{align*}
        \Delta W_s &=  \textrm{d}W_s / \textrm{d}t \, \Delta D\\
        \Delta W_f &=  \textrm{d}W_f / \textrm{d}t \, \Delta D
        \end{align*}
      \]

    .. NOTE::

        The original equations have been extended to include a term to model the costs
        of maintaining reproductive tissue mass as a fraction of foliage mass. These
        values can be set to zero to reproduce the predictions of the original T Model
        calculations. 




    Args:
        rho_s: Wood density of the PFT
        a_hd: Initial slope of the height/diameter relationship of the PFT
        h_max: Maximum height of the PFT
        lai: Leaf area index of the PFT
        ca_ratio: Crown area ratio of the PFT
        sla: Specific leaf area of the PFT
        zeta: The ratio of fine root mass to foliage area of the PFT
        npp: Net primary productivity of individuals
        turnover: Fine root and foliage turnover cost of individuals
        p_foliage_for_reproductive_tissue: Proportion of foliage mass that is
            reproductive tissue.
        reproductive_tissue_turnover: Reproductive tissue turnover cost of individuals
        dbh: Diameter at breast height of individuals
        stem_height: Stem height of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        size_args = {
            "npp": npp,
            "turnover": turnover,
            "reproductive_tissue_turnover": reproductive_tissue_turnover,
            "p_foliage_for_reproductive_tissue": p_foliage_for_reproductive_tissue,
            "dbh": dbh,
            "stem_height": stem_height,
        }
        _validate_demography_array_arguments(
            trait_args={
                "rho_s": rho_s,
                "a_hd": a_hd,
                "h_max": h_max,
                "lai": lai,
                "ca_ratio": ca_ratio,
                "sla": sla,
                "zeta": zeta,
            },
            size_args=size_args,
        )
        # Allow reproductive tissue terms to be zero
        size_args = {
            k: v
            for k, v in size_args.items()
            if k
            not in ("reproductive_tissue_turnover", "p_foliage_for_reproductive_tissue")
        }
        _enforce_positive_sizes(
            size_args=size_args, function_name="calculate_growth_increments"
        )

    # Rates of change in stem and foliar
    dWsdt = (
        np.pi
        / 8
        * rho_s
        * dbh
        * (a_hd * dbh * (1 - (stem_height / h_max)) + 2 * stem_height)
    )

    dWfdt = (
        lai
        * ((np.pi * ca_ratio) / (4 * a_hd))
        * (a_hd * dbh * (1 - stem_height / h_max) + stem_height)
        * ((1 + p_foliage_for_reproductive_tissue) / sla + zeta)
    )

    # Increment of diameter at breast height, ignoring potential zero divides resulting
    # from stems with zero DBH, which are then explicitly set to have Delta D of zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_d = _enforce_2D(
            np.where(
                dbh == 0,
                0,
                (npp - turnover - reproductive_tissue_turnover) / (dWsdt + dWfdt),
            )
        )

    return (delta_d, dWsdt * delta_d, dWfdt * delta_d)


@dataclass
class StemAllometry(PandasExporter, CohortMethods):
    """Calculate T Model allometric predictions across a set of stems.

    This method calculates predictions of stem allometries for stem height, crown area,
    crown fraction, stem mass, foliage mass and sapwood mass under the T Model
    :cite:`Li:2014bc`, given diameters at breast height for a set of plant functional
    traits.

    Args:
        stem_traits: An instance of :class:`~pyrealm.demography.flora.Flora` or
            :class:`~pyrealm.demography.flora.StemTraits`, providing plant functional
            trait data for a set of stems.
        at_dbh: An array of diameter at breast height values at which to predict stem
            allometry values.
        validate: Boolean flag to suppress argument validation
    """

    array_attrs: ClassVar[tuple[str, ...]] = (
        "dbh",
        "stem_height",
        "crown_area",
        "crown_fraction",
        "stem_mass",
        "foliage_mass",
        "reproductive_tissue_mass",
        "sapwood_mass",
        "crown_r0",
        "crown_z_max",
    )
    count_attr: ClassVar[str] = "_n_stems"

    # Init vars
    stem_traits: InitVar[Flora | StemTraits]
    """ An instance of :class:`~pyrealm.demography.flora.Flora` or 
    :class:`~pyrealm.demography.flora.StemTraits`, providing plant functional trait data
    for a set of stems."""
    at_dbh: InitVar[NDArray[np.float64]]
    """An array of diameter at breast height values at which to predict stem allometry 
    values."""
    validate: InitVar[bool] = True
    """Boolean flag to suppress argument validation."""

    # Post init allometry attributes
    dbh: NDArray[np.float64] = field(init=False)
    """Diameter at breast height (m)"""
    stem_height: NDArray[np.float64] = field(init=False)
    """Stem height (m)"""
    crown_area: NDArray[np.float64] = field(init=False)
    """Crown area (m2)"""
    crown_fraction: NDArray[np.float64] = field(init=False)
    """Vertical fraction of the stem covered by the crown (-)"""
    stem_mass: NDArray[np.float64] = field(init=False)
    """Stem mass (kg)"""
    foliage_mass: NDArray[np.float64] = field(init=False)
    """Foliage mass (kg)"""
    reproductive_tissue_mass: NDArray[np.float64] = field(init=False)
    """Reproductive tissue mass (kg)"""
    sapwood_mass: NDArray[np.float64] = field(init=False)
    """Sapwood mass (kg)"""
    crown_r0: NDArray[np.float64] = field(init=False)
    """Crown radius scaling factor (-)"""
    crown_z_max: NDArray[np.float64] = field(init=False)
    """Height of maximum crown radius (m)"""

    # Information attributes
    _n_pred: int = field(init=False)
    """The number of predictions per stem."""
    _n_stems: int = field(init=False)
    """The number of stems."""

    __experimental__ = True

    def __post_init__(
        self,
        stem_traits: Flora | StemTraits,
        at_dbh: NDArray[np.float64],
        validate: bool,
    ) -> None:
        """Populate the stem allometry attributes from the traits and size data."""

        warn_experimental("StemAllometry")

        # If validation is required, only need to perform validation once to check that
        # the at_dbh values are congruent with the stem_traits inputs. If they are, then
        # all the other allometry function inputs will be too.
        if validate:
            size_args = {"at_dbh": at_dbh}
            _validate_demography_array_arguments(
                trait_args={"h_max": stem_traits.h_max}, size_args=size_args
            )
            _enforce_positive_sizes(size_args=size_args, function_name="StemAllometry")

        self.stem_height = calculate_heights(
            h_max=stem_traits.h_max, a_hd=stem_traits.a_hd, dbh=at_dbh, validate=False
        )

        # Broadcast at_dbh to shape of stem height to get congruent shapes
        self.dbh = np.broadcast_to(at_dbh, self.stem_height.shape)

        self.crown_area = calculate_crown_areas(
            ca_ratio=stem_traits.ca_ratio,
            a_hd=stem_traits.a_hd,
            dbh=self.dbh,
            stem_height=self.stem_height,
            validate=False,
        )

        self.crown_fraction = calculate_crown_fractions(
            a_hd=stem_traits.a_hd,
            dbh=self.dbh,
            stem_height=self.stem_height,
            validate=False,
        )

        self.stem_mass = calculate_stem_masses(
            rho_s=stem_traits.rho_s,
            dbh=self.dbh,
            stem_height=self.stem_height,
            validate=False,
        )

        self.foliage_mass = calculate_foliage_masses(
            sla=stem_traits.sla,
            lai=stem_traits.lai,
            crown_area=self.crown_area,
            validate=False,
        )

        self.reproductive_tissue_mass = calculate_reproductive_tissue_mass(
            self.foliage_mass, stem_traits.p_foliage_for_reproductive_tissue
        )

        self.sapwood_mass = calculate_sapwood_masses(
            rho_s=stem_traits.rho_s,
            ca_ratio=stem_traits.ca_ratio,
            stem_height=self.stem_height,
            crown_area=self.crown_area,
            crown_fraction=self.crown_fraction,
            validate=False,
        )

        self.crown_r0 = calculate_crown_r0(
            q_m=stem_traits.q_m, crown_area=self.crown_area, validate=False
        )

        self.crown_z_max = calculate_crown_z_max(
            z_max_prop=stem_traits.z_max_prop,
            stem_height=self.stem_height,
            validate=False,
        )

        # Set the number of observations per stem as the length of axis 1
        self._n_pred = self.crown_z_max.shape[0]
        self._n_stems = stem_traits._n_stems

    def __repr__(self) -> str:
        return (
            f"StemAllometry: Prediction for {self._n_stems} stems "
            f"at {self._n_pred} DBH values."
        )


@dataclass()
class StemAllocation(PandasExporter):
    """Calculate T Model GPP allocation across a set of stems.

    This method calculates the predicted allocation of potential gross primary
    productivity (GPP) for stems under the T Model :cite:`Li:2014bc`, given a set of
    traits for those stems and the stem allometries given the stem size.

    Args:
        stem_traits: An instance of :class:`~pyrealm.demography.flora.Flora` or
            :class:`~pyrealm.demography.flora.StemTraits`, providing plant functional
            trait data for a set of stems.
        stem_allometry: An instance of
            :class:`~pyrealm.demography.tmodel.StemAllometry`
            providing the stem size data for which to calculate allocation.
        whole_crown_gpp: An array of GPP values available to a stem at which to model
            allocation (kg C).
        validate: Boolean flag to suppress argument validation
    """

    array_attrs: ClassVar[tuple[str, ...]] = (
        "whole_crown_gpp",
        "sapwood_respiration",
        "foliar_respiration",
        "fine_root_respiration",
        "reproductive_tissue_respiration",
        "npp",
        "foliage_turnover",
        "fine_root_turnover",
        "reproductive_tissue_turnover",
        "delta_dbh",
        "delta_stem_mass",
        "delta_foliage_mass",
    )

    # Init vars
    stem_traits: InitVar[Flora | StemTraits]
    """An instance of :class:`~pyrealm.demography.flora.Flora` or 
    :class:`~pyrealm.demography.flora.StemTraits`, providing plant functional trait data
    for a set of stems."""
    stem_allometry: InitVar[StemAllometry]
    """An instance of :class:`~pyrealm.demography.tmodel.StemAllometry`
    providing the stem size data for which to calculate allocation."""
    whole_crown_gpp: NDArray[np.float64]
    """An array of gross primary productivity values (kg C) across the whole of the
    crown of each stem to be allocated to respiration, turnover and growth."""
    validate: InitVar[bool] = True
    """ Boolean flag to suppress argument validation."""

    # Post init allometry attributes
    topslice_whole_crown_gpp: NDArray[np.float64] = field(init=False)
    """The available stem GPP after any topslicing (g C)"""
    sapwood_respiration: NDArray[np.float64] = field(init=False)
    """Allocation to sapwood respiration (g C)"""
    foliar_respiration: NDArray[np.float64] = field(init=False)
    """Allocation to foliar respiration (g C)"""
    reproductive_tissue_respiration: NDArray[np.float64] = field(init=False)
    """Allocation to reproductive tissue respiration (g C)"""
    fine_root_respiration: NDArray[np.float64] = field(init=False)
    """Allocation to fine root respiration (g C)"""
    gpp_topslice: NDArray[np.float64] = field(init=False)
    """GPP removed before allocation for various biological functions (g C)"""
    npp: NDArray[np.float64] = field(init=False)
    """Net primary productivity (g C)"""
    leaf_turnover: NDArray[np.float64] = field(init=False)
    """Allocation to leaf turnover (g C)"""
    fine_root_turnover: NDArray[np.float64] = field(init=False)
    """Allocation to fine root turnover"""
    reproductive_tissue_turnover: NDArray[np.float64] = field(init=False)
    """Allocation to reproductive tissue turnover (g C)"""
    delta_dbh: NDArray[np.float64] = field(init=False)
    """Predicted increase in stem diameter from growth allocation (m)"""
    delta_stem_mass: NDArray[np.float64] = field(init=False)
    """Predicted increase in stem mass from growth allocation (g C)"""
    delta_foliage_mass: NDArray[np.float64] = field(init=False)
    """Predicted increase in foliar mass from growth allocation (g C)"""

    # Information attributes
    _n_pred: int = field(init=False)
    """The number of predictions per stem."""
    _n_stems: int = field(init=False)
    """The number of stems."""

    __experimental__ = True

    def __post_init__(
        self,
        stem_traits: Flora | StemTraits,
        stem_allometry: StemAllometry,
        validate: bool,
    ) -> None:
        """Populate stem allocation attributes from the traits, allometry and GPP."""

        warn_experimental("StemAllocation")

        if validate:
            _validate_demography_array_arguments(
                trait_args={"h_max": stem_traits.h_max},
                size_args={"dbh": stem_allometry.dbh},
                at_size_args={"whole_crown_gpp": self.whole_crown_gpp},
            )

        # Broadcast potential GPP to match trait and size data outputs
        trait_size_shape = np.broadcast_shapes(
            stem_traits.h_max.shape, stem_allometry.dbh.shape
        )
        self.whole_crown_gpp = np.broadcast_to(self.whole_crown_gpp, trait_size_shape)

        self.gpp_topslice = calculate_gpp_topslice(
            gpp_topslice=stem_traits.gpp_topslice,
            whole_crown_gpp=self.whole_crown_gpp,
            validate=False,
        )

        # Topslice GPP
        self.topslice_whole_crown_gpp = self.whole_crown_gpp - self.gpp_topslice

        self.sapwood_respiration = calculate_sapwood_respiration(
            resp_s=stem_traits.resp_s,
            sapwood_mass=stem_allometry.sapwood_mass,
            validate=False,
        )

        self.foliar_respiration = calculate_foliar_respiration(
            resp_f=stem_traits.resp_f,
            whole_crown_gpp=self.topslice_whole_crown_gpp,
            validate=False,
        )

        self.reproductive_tissue_respiration = (
            calculate_reproductive_tissue_respiration(
                resp_rt=stem_traits.resp_rt,
                reproductive_tissue_mass=stem_allometry.reproductive_tissue_mass,
                validate=False,
            )
        )

        self.fine_root_respiration = calculate_fine_root_respiration(
            zeta=stem_traits.zeta,
            sla=stem_traits.sla,
            resp_r=stem_traits.resp_r,
            foliage_mass=stem_allometry.foliage_mass,
            validate=False,
        )

        self.npp = calculate_net_primary_productivity(
            yld=stem_traits.yld,
            whole_crown_gpp=self.topslice_whole_crown_gpp,
            foliar_respiration=self.foliar_respiration,
            fine_root_respiration=self.fine_root_respiration,
            sapwood_respiration=self.sapwood_respiration,
            reproductive_tissue_respiration=self.reproductive_tissue_respiration,
            validate=False,
        )

        self.foliage_turnover = calculate_foliage_turnover(
            tau_f=stem_traits.tau_f,
            foliage_mass=stem_allometry.foliage_mass,
            validate=False,
        )

        self.fine_root_turnover = calculate_fine_root_turnover(
            sla=stem_traits.sla,
            zeta=stem_traits.zeta,
            tau_r=stem_traits.tau_r,
            foliage_mass=stem_allometry.foliage_mass,
            validate=False,
        )

        self.reproductive_tissue_turnover = calculate_reproductive_tissue_turnover(
            reproductive_tissue_mass=stem_allometry.reproductive_tissue_mass,
            tau_rt=stem_traits.tau_rt,
            validate=False,
        )

        (self.delta_dbh, self.delta_stem_mass, self.delta_foliage_mass) = (
            calculate_growth_increments(
                rho_s=stem_traits.rho_s,
                a_hd=stem_traits.a_hd,
                h_max=stem_traits.h_max,
                lai=stem_traits.lai,
                ca_ratio=stem_traits.ca_ratio,
                sla=stem_traits.sla,
                zeta=stem_traits.zeta,
                npp=self.npp,
                turnover=self.foliage_turnover + self.fine_root_turnover,
                reproductive_tissue_turnover=self.reproductive_tissue_turnover,
                p_foliage_for_reproductive_tissue=stem_traits.p_foliage_for_reproductive_tissue,
                dbh=stem_allometry.dbh,
                stem_height=stem_allometry.stem_height,
                validate=False,
            )
        )

        # Set the number of observations per stem (one if dbh is 1D, otherwise size of
        # the first axis)
        if self.whole_crown_gpp.ndim == 1:
            self._n_pred = 1
        else:
            self._n_pred = self.whole_crown_gpp.shape[0]

        self._n_stems = stem_traits._n_stems

    def __repr__(self) -> str:
        return (
            f"StemAllocation: Prediction for {self._n_stems} stems "
            f"at {self._n_pred} observations."
        )
