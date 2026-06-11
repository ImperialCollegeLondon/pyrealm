"""The ``t_model`` module provides the basic scaling relationships of the T Model
:cite:`Li:2014bc`. This provides scaling relationships using the plant functional type
traits defined in the :mod:`~pyrealm.demography.flora` module and the diameter at breast
height of individual stems to define the stem geometry, masses, respiration and hence
calculate stem growth given net primary productivity. Note that
:attr:`~pyrealm.demography.tmodel.StemAllometry.stem_height` denotes the total tree
height, as used interchangeable in :cite:`Li:2014bc`, rather than just the height of the
trunk below the canopy.
"""  # noqa: D205

from typing import ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.demography.core import (
    CohortMethods,
    PandasExporter,
)
from pyrealm.demography_two.cohorts import Cohorts


def calculate_heights(
    h_max: NDArray[np.floating],
    a_hd: NDArray[np.floating],
    dbh: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return h_max * (1 - np.exp(-a_hd * dbh / h_max))


def calculate_dbh_from_height(
    h_max: NDArray[np.floating],
    a_hd: NDArray[np.floating],
    stem_height: NDArray[np.floating],
) -> NDArray[np.floating]:
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

    # The equation here blows up in a couple of ways:
    # - H > h_max leads to negative logs which generates np.nan with an invalid value
    #   warning. The np.nan here is what we want to happen, so the warning needs
    #   suppressing.
    # - H = h_max generates a divide by zero which returns inf with a warning. Here the
    #   answer should be h_max so that needs trapping.

    with np.errstate(divide="ignore", invalid="ignore"):
        return (h_max * np.log(h_max / (h_max - stem_height))) / a_hd


def calculate_crown_areas(
    ca_ratio: NDArray[np.floating],
    a_hd: NDArray[np.floating],
    dbh: NDArray[np.floating],
    stem_height: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return ((np.pi * ca_ratio) / (4 * a_hd)) * dbh * stem_height


def calculate_crown_fractions(
    a_hd: NDArray[np.floating],
    stem_height: NDArray[np.floating],
    dbh: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    # Calculate crown fraction
    return stem_height / (a_hd * dbh)


def calculate_stem_masses(
    rho_s: NDArray[np.floating],
    stem_height: NDArray[np.floating],
    dbh: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return (np.pi / 8) * rho_s * (dbh**2) * stem_height


def calculate_foliage_masses(
    sla: NDArray[np.floating],
    lai: NDArray[np.floating],
    crown_area: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return crown_area * lai * (1 / sla)


def calculate_fine_root_masses(
    lai: NDArray[np.floating],
    crown_area: NDArray[np.floating],
    zeta: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate foliage mass under the T Model.

    The fine root mass (:math:`W_{r}`) is calculated from the total area of foliage -
    the product of the crown area (:math:`A_{c}`) and leaf area index (:math:`L`) - and
    the ratio of fine root mass to leaf area (:math:`zeta`).

    .. math::

        W_r = A_c L \zeta

    Args:
        lai: Leaf area index of the PFT
        crown_area: Crown area of individuals
        zeta: The ratio of fine root mass to foliage area of the PFT.
    """

    return crown_area * lai * zeta


def calculate_sapwood_masses(
    rho_s: NDArray[np.floating],
    ca_ratio: NDArray[np.floating],
    stem_height: NDArray[np.floating],
    crown_area: NDArray[np.floating],
    crown_fraction: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return crown_area * rho_s * stem_height * (1 - crown_fraction / 2) / ca_ratio


def calculate_crown_z_max(
    z_max_prop: NDArray[np.floating],
    stem_height: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return stem_height * z_max_prop


def calculate_crown_r0(
    q_m: NDArray[np.floating],
    crown_area: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    # Scaling factor to give expected A_c (crown area) at
    # z_m (height of maximum crown radius)
    return 1 / q_m * np.sqrt(crown_area / np.pi)


def calculate_whole_crown_gpp(
    potential_gpp: NDArray[np.floating],
    crown_area: NDArray[np.floating],
    par_ext: NDArray[np.floating],
    lai: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return potential_gpp * crown_area * (1 - np.exp(-(par_ext * lai)))


def calculate_sapwood_respiration(
    resp_s: NDArray[np.floating],
    sapwood_mass: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate sapwood respiration.

    Calculates the total sapwood respiration (:math:`R_{\cdot s}`) given the individual
    sapwood mass (:math:`W_{\cdot s}`) and the sapwood respiration rate of the plant
    functional type (:math:`r_{s}`) :cite:p:`{see Equation 13, }Li:2014bc`.

    .. math::
         R_{\cdot s} = W_{\cdot s} \, r_s

    Args:
        resp_s: The sapwood respiration rate
        sapwood_mass: The individual sapwood mass
    """

    return sapwood_mass * resp_s


def calculate_foliage_respiration(
    resp_f: NDArray[np.floating],
    whole_crown_gpp: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return whole_crown_gpp * resp_f


def calculate_gpp_topslice(
    gpp_topslice: NDArray[np.floating],
    whole_crown_gpp: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """
    return whole_crown_gpp * gpp_topslice


def calculate_reproductive_tissue_respiration(
    resp_rt: NDArray[np.floating],
    reproductive_tissue_mass: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return reproductive_tissue_mass * resp_rt


def calculate_fine_root_respiration(
    fine_root_mass: NDArray[np.floating],
    resp_r: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate fine root respiration.

    Calculates the total fine root respiration (:math:`R_{r}`) given fine root mass
    (:math:`W_r`) the fine root respiration rate (:math:`r_r`):

    .. math::
         R_{r} = W_r r_r

    Equation 13 of :cite:`Li:2014bc` gives this calculation as:

    .. math::
         R_{r} = \zeta \sigma W_f r_r,

    given the individual foliage mass (:math:`W_f`), the ratio of fine root mass to
    foliage area (:math:`\zeta`) and the specific leaf area (:math:`\sigma`), which can
    be simplified to the equation here given :math: `W_f = (A_c L) / \sigma`: and
    :math:`W_r = \zeta A_c L` (see :func:`calculate_fine_root_masses`).

    Args:
        fine_root_mass: The individual fine root mass.
        resp_r: The respiration rate of fine roots of the PFT.
    """

    return fine_root_mass * resp_r


def calculate_net_primary_productivity(
    yld: NDArray[np.floating],
    whole_crown_gpp: NDArray[np.floating],
    foliage_respiration: NDArray[np.floating],
    fine_root_respiration: NDArray[np.floating],
    sapwood_respiration: NDArray[np.floating],
    reproductive_tissue_respiration: NDArray[np.floating],
) -> NDArray[np.floating]:
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
        foliage_respiration: The total foliar respiration.
        fine_root_respiration: The total fine root respiration
        sapwood_respiration: The total sapwood respiration.
        reproductive_tissue_respiration: The total reproductive tissue respiration.
    """

    return yld * (
        whole_crown_gpp
        - foliage_respiration
        - fine_root_respiration
        - sapwood_respiration
        - reproductive_tissue_respiration
    )


def calculate_foliage_turnover(
    tau_f: NDArray[np.floating],
    foliage_mass: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return foliage_mass * (1 / tau_f)


def calculate_fine_root_turnover(
    tau_r: NDArray[np.floating],
    fine_root_mass: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate turnover costs.

    This function calculates the costs associated with the turnover of fine roots. This
    is calculated from the total fine root mass of individuals (:math:`W_r`) and the
    turnover time of fine roots (:math:`\tau_r`) of the plant functional type.

    .. math::

        T = \frac{W_r}{\tau_r}

    Equation 15 of :cite:`Li:2014bc` gives this as:

        T = W_f \left(\frac{ \sigma \zeta}{\tau_r} \right),

    given the foliage mass of individuals (:math:`W_f`), the specific leaf area
    (:math:`\sigma`) and fine root mass to foliage area ratio (:math:`\zeta`), which can
    be simplified to the equation here given :math: `W_f = (A_c L) / \sigma`: and
    :math:`W_r = \zeta A_c L` (see :func:`calculate_fine_root_masses`).


    Args:
        tau_r: The turnover time of fine roots
        fine_root_mass: The fine root mass
    """

    return fine_root_mass / tau_r


def calculate_reproductive_tissue_turnover(
    reproductive_tissue_mass: NDArray[np.floating],
    tau_rt: NDArray[np.floating],
) -> NDArray[np.floating]:
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
    """

    return reproductive_tissue_mass * (1 / tau_rt)


def calculate_reproductive_tissue_mass(
    foliage_mass: NDArray[np.floating],
    p_foliage_for_reproductive_tissue: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate reproductive tissue mass.

    This function calculates the mass of reproductive tissue (:math:`m_{rt}`) as a fixed
    proportion of the total foliage mass (:math:`W_f`) of individuals.

    .. math::

        m_{rt} = p_{f_{rt}} W_f

    Args:
        foliage_mass: The foliage mass
        p_foliage_for_reproductive_tissue: The proportion of foliage mass that is
            reproductive tissue
    """

    return p_foliage_for_reproductive_tissue * foliage_mass


def calculate_growth_increments(
    rho_s: NDArray[np.floating],
    a_hd: NDArray[np.floating],
    h_max: NDArray[np.floating],
    lai: NDArray[np.floating],
    ca_ratio: NDArray[np.floating],
    sla: NDArray[np.floating],
    zeta: NDArray[np.floating],
    npp: NDArray[np.floating],
    turnover: NDArray[np.floating],
    reproductive_tissue_turnover: NDArray[np.floating],
    p_foliage_for_reproductive_tissue: NDArray[np.floating],
    dbh: NDArray[np.floating],
    stem_height: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    r"""Calculate growth increments.

    Given an estimate of net primary productivity (NPP, :math:`P_{net}`), less
    associated turnover costs (:math:`T`), the remaining productivity can be allocated
    to growth and hence estimate resulting increments :cite:`Li:2014bc` in:
    
    * the stem diameter (:math:`\Delta D`),
    * the stem mass (:math:`\Delta W_s`), 
    * the foliar mass (:math:`\Delta W_f`), and
    * the fine root mass (:math:`\Delta W_r`).
    
    The stem diameter increment can be calculated using the available productivity for
    growth and the rates of change in stem mass (:math:`\textrm{d}W_s / \textrm{d}t`)
    and in the combined foliage and fine root masses (:math:`\textrm{d}W_fr /
    \textrm{d}t`):  

    .. math::

        \Delta D = \frac{P_{net} - T}{ \textrm{d}W_s / \textrm{d}t  +
             \textrm{d}W_fr / \textrm{d}t}

    The rates of change in stem and foliar mass can be calculated as:

    .. math::
      :nowrap:

      \[
        \begin{align*}
            \textrm{d}W_s / \textrm{d}t &= \frac{\pi}{8} \rho_s D
                \left(a D \left(1 - \frac{H}{H_{m}} + 2 H \right) \right) \\

            \textrm{d}W_fr / \textrm{d}t &= L \frac{\pi c}{4 a} \left(a D \left( 1 -
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

    The resulting incremental changes in stem mass and foliage plus fine root masses can
    then be calculated as:

    .. math::
      :nowrap:

      \[
        \begin{align*}
        \Delta W_s &=  \textrm{d}W_s / \textrm{d}t \, \Delta D\\
        \Delta W_fr &=  \textrm{d}W_fr / \textrm{d}t \, \Delta D
        \end{align*}
      \]

    Note that :cite:`Li:2014bc` use ':math:`W_f`' to denote the increment in both
    foliage and fine root mass, as fine root mass is estimated as a function of foliage
    area through the specific leaf area (:math:`\sigma`) and  ratio of fine root mass to
    leaf area (:math:`\zeta`). Here we use :math:`W_fr` to indicate the combined
    increments and partition the final increments into foliage and fine root components
    as:

    .. math::
      :nowrap:

      \[
        \begin{align*}

        \Delta W_f &= \Delta W_fr /( 1 + \sigma \zeta)
        \Delta W_r &= \Delta W_fr - \Delta W_f
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
    """

    # Rates of change in stem and foliage + fine root mass
    dWsdt = (
        np.pi
        / 8
        * rho_s
        * dbh
        * (a_hd * dbh * (1 - (stem_height / h_max)) + 2 * stem_height)
    )

    # This equation includes terms for the rate of change in fine root mass, which is
    # estimated alongside rate of change in foliage mass in the model as
    # (1 + sigma + zeta) dWfdt (Eqn 15)
    dWfrdt = (
        lai
        * ((np.pi * ca_ratio) / (4 * a_hd))
        * (a_hd * dbh * (1 - stem_height / h_max) + stem_height)
        * ((1 + p_foliage_for_reproductive_tissue) / sla + zeta)
    )

    # Increment of diameter at breast height, ignoring potential zero divides resulting
    # from stems with zero DBH, which are then explicitly set to have Delta D of zero.
    # TODO - can we remove this check since Cohorts and Allocation enforce DBH > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_d = np.where(
            dbh == 0,
            0,
            (npp - turnover - reproductive_tissue_turnover) / (dWsdt + dWfrdt),
        )

    # Partition delta Wfr into delta Wf and delta Wr using (1 + sigma.zeta)
    fine_root_foliage_factor = 1 + sla * zeta
    delta_Wfr = dWfrdt * delta_d
    delta_Wf = delta_Wfr / fine_root_foliage_factor
    delta_Wr = delta_Wfr - delta_Wf

    return (delta_d, dWsdt * delta_d, delta_Wf, delta_Wr)


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
    """

    array_attrs: ClassVar[tuple[str, ...]] = (
        "dbh",
        "stem_height",
        "crown_area",
        "crown_fraction",
        "stem_mass",
        "foliage_mass",
        "fine_root_mass",
        "reproductive_tissue_mass",
        "sapwood_mass",
        "crown_r0",
        "crown_z_max",
    )
    count_attr: ClassVar[str] = "_n_stems"

    __experimental__ = True

    def __init__(
        self,
        cohorts: Cohorts,
        at_dbh: NDArray[np.floating] | None = None,
    ) -> None:
        """Populate the stem allometry attributes from the traits and size data."""

        warn_experimental("StemAllometry")

        self.at_dbh: NDArray[np.floating]
        """An array of diameter at breast height values at which to predict stem
        allometry values."""

        # Allometry attributes
        self.stem_height: NDArray[np.floating]
        """Stem height (m)"""
        self.crown_area: NDArray[np.floating]
        """Crown area (m2)"""
        self.crown_fraction: NDArray[np.floating]
        """Vertical fraction of the stem covered by the crown (-)"""
        self.stem_mass: NDArray[np.floating]
        """Stem mass (kg)"""
        self.foliage_mass: NDArray[np.floating]
        """Foliage mass (kg)"""
        self.fine_root_mass: NDArray[np.floating]
        """Fine root mass (kg)"""
        self.reproductive_tissue_mass: NDArray[np.floating]
        """Reproductive tissue mass (kg)"""
        self.sapwood_mass: NDArray[np.floating]
        """Sapwood mass (kg)"""
        self.crown_r0: NDArray[np.floating]
        """Crown radius scaling factor (-)"""
        self.crown_z_max: NDArray[np.floating]
        """Height of maximum crown radius (m)"""

        # Populate DBH values for calculating allometry. The CohortData code already
        # enforces positive DBH, so only need to check at_dbh.
        if at_dbh is None:
            # If no at_dbh is provided, use the dbh values from the cohorts.
            self.at_dbh = cohorts.cohorts["dbh_value"].to_numpy()
        else:
            # Validate 1D array and transpose to a column array.
            if not (isinstance(at_dbh, np.ndarray) and at_dbh.ndim == 1):
                raise ValueError("The at_dbh value must be a 1D numpy array.")
            if np.any(at_dbh <= 0):
                raise ValueError("Values in at_dbh must be greater than zero.")
            self.at_dbh = at_dbh[None, :].T

        self.stem_height = calculate_heights(
            h_max=cohorts.cohorts["h_max"].to_numpy(),
            a_hd=cohorts.cohorts["a_hd"].to_numpy(),
            dbh=self.at_dbh,
        )

        self.crown_area = calculate_crown_areas(
            ca_ratio=cohorts.cohorts["ca_ratio"].to_numpy(),
            a_hd=cohorts.cohorts["a_hd"].to_numpy(),
            dbh=self.at_dbh,
            stem_height=self.stem_height,
        )

        self.crown_fraction = calculate_crown_fractions(
            a_hd=cohorts.cohorts["a_hd"].to_numpy(),
            dbh=self.at_dbh,
            stem_height=self.stem_height,
        )

        self.stem_mass = calculate_stem_masses(
            rho_s=cohorts.cohorts["rho_s"].to_numpy(),
            dbh=self.at_dbh,
            stem_height=self.stem_height,
        )

        self.foliage_mass = calculate_foliage_masses(
            sla=cohorts.cohorts["sla"].to_numpy(),
            lai=cohorts.cohorts["lai"].to_numpy(),
            crown_area=self.crown_area,
        )

        self.fine_root_mass = calculate_fine_root_masses(
            zeta=cohorts.cohorts["zeta"].to_numpy(),
            lai=cohorts.cohorts["lai"].to_numpy(),
            crown_area=self.crown_area,
        )

        self.reproductive_tissue_mass = calculate_reproductive_tissue_mass(
            self.foliage_mass,
            cohorts.cohorts["p_foliage_for_reproductive_tissue"].to_numpy(),
        )

        self.sapwood_mass = calculate_sapwood_masses(
            rho_s=cohorts.cohorts["rho_s"].to_numpy(),
            ca_ratio=cohorts.cohorts["ca_ratio"].to_numpy(),
            stem_height=self.stem_height,
            crown_area=self.crown_area,
            crown_fraction=self.crown_fraction,
        )

        self.crown_r0 = calculate_crown_r0(
            q_m=cohorts.cohorts["q_m"].to_numpy(),
            crown_area=self.crown_area,
        )

        self.crown_z_max = calculate_crown_z_max(
            z_max_prop=cohorts.cohorts["z_max_prop"].to_numpy(),
            stem_height=self.stem_height,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the allometries as a dataframe."""
        if self.at_dbh.ndim > 1:
            raise ValueError(
                "Allometries calculated for multiple DBH values, "
                "cannot convert to data frame"
            )

        return pd.DataFrame(
            {
                "stem_height": self.stem_height,
                "crown_area": self.crown_area,
                "crown_fraction": self.crown_fraction,
                "stem_mass": self.stem_mass,
                "foliage_mass": self.foliage_mass,
                "fine_root_mass": self.fine_root_mass,
                "reproductive_tissue_mass": self.reproductive_tissue_mass,
                "sapwood_mass": self.sapwood_mass,
                "crown_r0": self.crown_r0,
                "crown_z_max": self.crown_z_max,
            }
        )

    #     # Set the number of observations per stem as the length of axis 1
    #     self._n_pred = self.crown_z_max.shape[0]
    #     self._n_stems = cohorts.cohorts["_n_stems"].to_numpy()

    # def __repr__(self) -> str:
    #     return (
    #         f"StemAllometry: Prediction for {self._n_stems} stems "
    #         f"at {self._n_pred} DBH values."
    #     )


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
    """

    array_attrs: ClassVar[tuple[str, ...]] = (
        "whole_crown_gpp",
        "sapwood_respiration",
        "foliage_respiration",
        "fine_root_respiration",
        "reproductive_tissue_respiration",
        "npp",
        "foliage_turnover",
        "fine_root_turnover",
        "reproductive_tissue_turnover",
        "delta_dbh",
        "delta_stem_mass",
        "delta_foliage_mass",
        "delta_fine_root_mass",
    )

    __experimental__ = True

    def __init__(
        self,
        cohorts: Cohorts,
        allometry: StemAllometry,
        whole_crown_gpp: NDArray[np.floating],
    ) -> None:
        """Calculate allocation of GPP for cohorts."""

        warn_experimental("StemAllocation")

        self.whole_crown_gpp: NDArray[np.floating]
        """An array of gross primary productivity values (kg C) across the whole of the
        self.crown of each stem to be allocated to respiration, turnover and growth."""
        self.topslice_whole_crown_gpp: NDArray[np.floating]
        """The available stem GPP after any topslicing (g C)"""
        self.sapwood_respiration: NDArray[np.floating]
        """Allocation to sapwood respiration (g C)"""
        self.foliage_respiration: NDArray[np.floating]
        """Allocation to foliar respiration (g C)"""
        self.reproductive_tissue_respiration: NDArray[np.floating]
        """Allocation to reproductive tissue respiration (g C)"""
        self.fine_root_respiration: NDArray[np.floating]
        """Allocation to fine root respiration (g C)"""
        self.gpp_topslice: NDArray[np.floating]
        """GPP removed before allocation for various biological functions (g C)"""
        self.npp: NDArray[np.floating]
        """Net primary productivity (g C)"""
        self.foliage_turnover: NDArray[np.floating]
        """Allocation to leaf turnover (g C)"""
        self.fine_root_turnover: NDArray[np.floating]
        """Allocation to fine root turnover"""
        self.reproductive_tissue_turnover: NDArray[np.floating]
        """Allocation to reproductive tissue turnover (g C)"""
        self.delta_dbh: NDArray[np.floating]
        """Predicted increase in stem diameter from growth allocation (m)"""
        self.delta_stem_mass: NDArray[np.floating]
        """Predicted increase in stem mass from growth allocation (g C)"""
        self.delta_foliage_mass: NDArray[np.floating]
        """Predicted increase in foliar mass from growth allocation (g C)"""
        self.delta_fine_root_mass: NDArray[np.floating]
        """Predicted increase in fine root mass from growth allocation (g C)"""

        self.whole_crown_gpp = whole_crown_gpp

        self.gpp_topslice = calculate_gpp_topslice(
            gpp_topslice=cohorts.cohorts["gpp_topslice"].to_numpy(),
            whole_crown_gpp=self.whole_crown_gpp,
        )

        # Topslice GPP
        self.topslice_whole_crown_gpp = self.whole_crown_gpp - self.gpp_topslice

        # Calculate respiration terms
        self.sapwood_respiration = calculate_sapwood_respiration(
            resp_s=cohorts.cohorts["resp_s"].to_numpy(),
            sapwood_mass=allometry.sapwood_mass,
        )

        self.foliage_respiration = calculate_foliage_respiration(
            resp_f=cohorts.cohorts["resp_f"].to_numpy(),
            whole_crown_gpp=self.topslice_whole_crown_gpp,
        )

        self.reproductive_tissue_respiration = (
            calculate_reproductive_tissue_respiration(
                resp_rt=cohorts.cohorts["resp_rt"].to_numpy(),
                reproductive_tissue_mass=allometry.reproductive_tissue_mass,
            )
        )

        self.fine_root_respiration = calculate_fine_root_respiration(
            resp_r=cohorts.cohorts["resp_r"].to_numpy(),
            fine_root_mass=allometry.fine_root_mass,
        )

        # Calculate NPP given losses to yield and respiration costs
        self.npp = calculate_net_primary_productivity(
            yld=cohorts.cohorts["yld"].to_numpy(),
            whole_crown_gpp=self.topslice_whole_crown_gpp,
            foliage_respiration=self.foliage_respiration,
            fine_root_respiration=self.fine_root_respiration,
            sapwood_respiration=self.sapwood_respiration,
            reproductive_tissue_respiration=self.reproductive_tissue_respiration,
        )

        # Calculate turnover costs
        self.foliage_turnover = calculate_foliage_turnover(
            tau_f=cohorts.cohorts["tau_f"].to_numpy(),
            foliage_mass=allometry.foliage_mass,
        )

        self.fine_root_turnover = calculate_fine_root_turnover(
            tau_r=cohorts.cohorts["tau_r"].to_numpy(),
            fine_root_mass=allometry.fine_root_mass,
        )

        self.reproductive_tissue_turnover = calculate_reproductive_tissue_turnover(
            reproductive_tissue_mass=allometry.reproductive_tissue_mass,
            tau_rt=cohorts.cohorts["tau_rt"].to_numpy(),
        )

        # Calculate resulting growth increments given NPP and turnover costs.
        (
            self.delta_dbh,
            self.delta_stem_mass,
            self.delta_foliage_mass,
            self.delta_fine_root_mass,
        ) = calculate_growth_increments(
            rho_s=cohorts.cohorts["rho_s"].to_numpy(),
            a_hd=cohorts.cohorts["a_hd"].to_numpy(),
            h_max=cohorts.cohorts["h_max"].to_numpy(),
            lai=cohorts.cohorts["lai"].to_numpy(),
            ca_ratio=cohorts.cohorts["ca_ratio"].to_numpy(),
            sla=cohorts.cohorts["sla"].to_numpy(),
            zeta=cohorts.cohorts["zeta"].to_numpy(),
            npp=self.npp,
            turnover=self.foliage_turnover + self.fine_root_turnover,
            reproductive_tissue_turnover=self.reproductive_tissue_turnover,
            p_foliage_for_reproductive_tissue=cohorts.cohorts[
                "p_foliage_for_reproductive_tissue"
            ].to_numpy(),
            dbh=allometry.at_dbh,
            stem_height=allometry.stem_height,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the allometries as a dataframe."""
        if self.whole_crown_gpp.ndim > 1:
            raise ValueError(
                "Allocations calculated for multiple DBH or GPP values, "
                "cannot convert to data frame"
            )

        return pd.DataFrame(
            {
                "whole_crown_gpp": self.whole_crown_gpp,
                "topslice_whole_crown_gpp": self.topslice_whole_crown_gpp,
                "sapwood_respiration": self.sapwood_respiration,
                "foliage_respiration": self.foliage_respiration,
                "reproductive_tissue_respiration": self.reproductive_tissue_respiration,
                "fine_root_respiration": self.fine_root_respiration,
                "gpp_topslice": self.gpp_topslice,
                "npp": self.npp,
                "foliage_turnover": self.foliage_turnover,
                "fine_root_turnover": self.fine_root_turnover,
                "reproductive_tissue_turnover": self.reproductive_tissue_turnover,
                "delta_dbh": self.delta_dbh,
                "delta_stem_mass": self.delta_stem_mass,
                "delta_foliage_mass": self.delta_foliage_mass,
                "delta_fine_root_mass": self.delta_fine_root_mass,
            }
        )

    # def __repr__(self) -> str:
    #     return (
    #         f"StemAllocation: Prediction for {self._n_stems} stems "
    #         f"at {self._n_pred} observations."
    #     )
