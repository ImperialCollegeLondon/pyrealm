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
from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.demography.cohorts import Cohorts
from pyrealm.demography.core import ToDataFrameMixin


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
) -> NDArray[np.floating]:
    r"""Calculate net primary productivity.

    The net primary productivity (NPP, :math:`P_{net}`) is calculated as a plant
    functional type specific yield proportion (:math:`y`) of the total GPP (:math:`P`)
    for the individual minus respiration (:math:`R_m`), as the sum of the respiration
    costs for foliage  (:math:`R_f`), fine roots  (:math:`R_r`) and sapwood
    (:math:`R_s`).

    .. math::
        P_{net} = y (P - R_m) = y (P - W_{\cdot s} r_s - \zeta \sigma W_f r_r - W_f r_f)

    Note that this differs from Equation 13 of :cite:t:`Li:2014bc`, which does not
    include a term for foliar respiration. :cite:t:`Li:2014bc` remove foliar respiration
    as a fixed proportion of potential GPP as the first step in their calculations. The
    approach here is equivalent but allows the foliar respiration to vary between plant
    functional types.

    Args:
        yld: The yield proportion.
        whole_crown_gpp: The total GPP for the crown.
        foliage_respiration: The total foliar respiration.
        fine_root_respiration: The total fine root respiration
        sapwood_respiration: The total sapwood respiration.
    """

    return yld * (
        whole_crown_gpp
        - foliage_respiration
        - fine_root_respiration
        - sapwood_respiration
    )


def calculate_foliage_turnover(
    tau_f: NDArray[np.floating],
    foliage_mass: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate foliage turnover.

    This function calculates the carbon mass of foliage turnover. This is calculated
    from the total foliage mass of individuals (:math:`W_f`), and the turnover times of
    foliage (:math:`\tau_f`) of the plant functional type :cite:p:`{see Equation 15,
    }Li:2014bc`.

    .. math::

        T = W_f \left( \frac{1}{\tau_f} \right)

    Args:
        tau_f: The turnover time of foliage
        foliage_mass: The foliage mass
    """

    return foliage_mass / tau_f


def calculate_branch_turnover(
    tau_b: NDArray[np.floating],
    stem_mass: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate stem turnover.

    This function calculates the carbon mass of branch turnover, representing branch
    fall and other woody tissue losses. This is calculated from the total stem mass of
    individuals (:math:`W_s`), and the stem turnover rate (:math:`\tau_b`) for the
    plant functional type.

    .. math::

        T = W_s * \tau_b

    NOTE::

        Th :math:`\tau_b` term is not present in :cite:t:`{see Equation 15, }Li:2014bc`
        and is added in pyrealm. It defaults to infinity to duplicate the calculations
        of the original model, which do not include branch turnover.

    Args:
        tau_b: The branch turnover rate
        stem_mass: The stem mass
    """

    # This handles the default infinite turnover value because X / Inf = 0 for all X.
    return stem_mass / tau_b


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


def calculate_growth_increments(
    rho_s: NDArray[np.floating],
    a_hd: NDArray[np.floating],
    h_max: NDArray[np.floating],
    lai: NDArray[np.floating],
    ca_ratio: NDArray[np.floating],
    sla: NDArray[np.floating],
    zeta: NDArray[np.floating],
    biomass_production: NDArray[np.floating],
    turnover: NDArray[np.floating],
    dbh: NDArray[np.floating],
    stem_height: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    r"""Calculate growth increments.

    This function calculates growth increments for stems. Under the T Model
    :cite:`Li:2014bc`, estimated biomass production (:math:`B`) can be partitioned into
    turnover costs (:math:`T`) and carbon available for allocation to biomass increments
    in:
    
    * the stem diameter (:math:`\Delta D`),
    * the stem mass (:math:`\Delta W_s`), 
    * the foliar mass (:math:`\Delta W_f`), and
    * the fine root mass (:math:`\Delta W_r`).

    The T Model does not include the allocation of NPP to carbon costs outside of growth
    and turnover, and so uses NPP directly as biomass production. Predicted NPP could be
    decreased to capture allocation to other processes, such as VOC emissions,
    non-structural carbohydrates or soil exudates, that are not currently modelled
    within ``pyrealm``.

    The stem diameter increment can be calculated using the available productivity for
    growth and the rates of change in stem mass (:math:`\textrm{d}W_s / \textrm{d}t`)
    and in the combined foliage and fine root masses (:math:`\textrm{d}W_fr /
    \textrm{d}t`):  

    .. math::

        \Delta D = \frac{B - T}{ \textrm{d}W_s / \textrm{d}t  +
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


    Args:
        rho_s: Wood density of the PFT
        a_hd: Initial slope of the height/diameter relationship of the PFT
        h_max: Maximum height of the PFT
        lai: Leaf area index of the PFT
        ca_ratio: Crown area ratio of the PFT
        sla: Specific leaf area of the PFT
        zeta: The ratio of fine root mass to foliage area of the PFT
        biomass_production: The biomass production of individuals
        turnover: Fine root and foliage turnover cost of individuals
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
        * (1 / sla + zeta)
    )

    # Increment of diameter at breast height, ignoring potential zero divides resulting
    # from stems with zero DBH, which are then explicitly set to have Delta D of zero.
    # TODO - can we remove this check since Cohorts and Allocation enforce DBH > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_d = np.where(
            dbh == 0,
            0,
            (biomass_production - turnover) / (dWsdt + dWfrdt),
        )

    # Partition delta Wfr into delta Wf and delta Wr using (1 + sigma.zeta)
    fine_root_foliage_factor = 1 + sla * zeta
    delta_Wfr = dWfrdt * delta_d
    delta_Wf = delta_Wfr / fine_root_foliage_factor
    delta_Wr = delta_Wfr - delta_Wf

    return (delta_d, dWsdt * delta_d, delta_Wf, delta_Wr)


class StemAllometry(ToDataFrameMixin):
    """Calculate T Model allometric predictions across a set of stems.

    This method calculates predictions of stem allometries for stem height, crown area,
    crown fraction, stem mass, foliage mass and sapwood mass under the T Model
    :cite:`Li:2014bc`, given diameters at breast height (DBH) for a set of plant
    cohorts.

    The default is to calculate the expected allometry for the DBH values specified in
    the cohort data and the predictions will be 1D arrays providing the prediction for
    each cohort.

    Alternatively, the class can be used to generate a stem allometry profile for the
    cohort PFTs at a range of DBH values. The ``at_dbh`` argument is used to provide 1D
    array of DBH values and the class will then generate a prediction for each cohort
    PFT at each stem diameter. The class prediction attributes are then 2D arrays with
    shape `(n_at_dbh, n_cohorts)`.

    The ``to_dataframe`` method can be used to export the predictions as a
    data frame, flattening 2D predictions if ``at_dbh`` is used.

    Args:
        cohorts: An instance of :class:`~pyrealm.demography.cohorts.Cohorts`.
        at_dbh: An optional array of DBH values used to provide a profile of allometry
            predictions.
    """

    _array_attrs: ClassVar[tuple[str, ...]] = (
        "cohort_id",
        "dbh",
        "stem_height",
        "crown_area",
        "crown_fraction",
        "stem_mass",
        "foliage_mass",
        "fine_root_mass",
        "sapwood_mass",
        "crown_r0",
        "crown_z_max",
    )

    __experimental__ = True

    def __init__(
        self,
        cohorts: Cohorts,
        at_dbh: NDArray[np.floating] | None = None,
    ) -> None:
        """Populate the stem allometry attributes from the traits and size data."""

        warn_experimental("StemAllometry")

        self._at_dbh_set: bool
        """An flag recording whether at_dbh was passed."""
        self._ndims: int
        """An integer giving the dimensionality of the predictions."""

        # Allometry attributes
        self.cohort_id: NDArray[np.generic]
        """An array of the cohort ID for each prediction."""
        self.dbh: NDArray[np.floating]
        """The diameter at breast height (m)"""
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
            self.dbh = cohorts.dbh_value.to_numpy()
            self.cohort_id = cohorts.cohort_id.to_numpy()
            self._at_dbh_set = False
            self._ndims = 1
        else:
            # Validate at_dbh
            if not (isinstance(at_dbh, np.ndarray) and at_dbh.ndim == 1):
                raise ValueError("The at_dbh value must be a 1D numpy array.")
            if np.any(at_dbh <= 0):
                raise ValueError("Values in at_dbh must be greater than zero.")

            # Broadcast DBH and cohort IDs to their outer product,
            self.dbh, self.cohort_id = np.broadcast_arrays(
                at_dbh[:, None], cohorts.cohort_id.to_numpy()
            )
            self._at_dbh_set = True
            self._ndims = 2

        self._cohort_id = cohorts.cohort_id.to_numpy()

        self.stem_height = calculate_heights(
            h_max=cohorts.h_max.to_numpy(),
            a_hd=cohorts.a_hd.to_numpy(),
            dbh=self.dbh,
        )

        self.crown_area = calculate_crown_areas(
            ca_ratio=cohorts.ca_ratio.to_numpy(),
            a_hd=cohorts.a_hd.to_numpy(),
            dbh=self.dbh,
            stem_height=self.stem_height,
        )

        self.crown_fraction = calculate_crown_fractions(
            a_hd=cohorts.a_hd.to_numpy(),
            dbh=self.dbh,
            stem_height=self.stem_height,
        )

        self.stem_mass = calculate_stem_masses(
            rho_s=cohorts.rho_s.to_numpy(),
            dbh=self.dbh,
            stem_height=self.stem_height,
        )

        self.foliage_mass = calculate_foliage_masses(
            sla=cohorts.sla.to_numpy(),
            lai=cohorts.lai.to_numpy(),
            crown_area=self.crown_area,
        )

        self.fine_root_mass = calculate_fine_root_masses(
            zeta=cohorts.zeta.to_numpy(),
            lai=cohorts.lai.to_numpy(),
            crown_area=self.crown_area,
        )

        self.sapwood_mass = calculate_sapwood_masses(
            rho_s=cohorts.rho_s.to_numpy(),
            ca_ratio=cohorts.ca_ratio.to_numpy(),
            stem_height=self.stem_height,
            crown_area=self.crown_area,
            crown_fraction=self.crown_fraction,
        )

        self.crown_r0 = calculate_crown_r0(
            q_m=cohorts.q_m.to_numpy(),
            crown_area=self.crown_area,
        )

        self.crown_z_max = calculate_crown_z_max(
            z_max_prop=cohorts.z_max_prop.to_numpy(),
            stem_height=self.stem_height,
        )

    def __repr__(self) -> str:
        if self._at_dbh_set:
            return (
                "StemAllometry: Allometry predictions for {1} cohorts "
                "at {0} DBH values."
            ).format(*self.dbh.shape)

        return "StemAllometry: Allometry predictions for {} cohorts.".format(
            *self.stem_height.shape
        )


class StemAllocation(ToDataFrameMixin):
    """Calculate GPP allocation for stems.

    This method calculates the predicted GPP allocations of potential gross primary
    productivity (GPP) for stems under the T Model :cite:`Li:2014bc`, given a set of
    cohorts and stem allometry predictions for those cohorts.

    Allocation from GPP estimates are handled in two ways:

    1. In the standard mode, provided GPP estimates are mapped onto the provided
       allometry following standard array broadcasting. For example, if the allometry
       provides data for three cohorts, then GPP values could be a scalar array (shape
       `(1,)`) or provide and estimate for each cohort (shape `(3,)`. If the allometry
       estimates used ``at_dbh`` to estimate allometry for four DBH sizes, then GPP
       could again be scalar or per cohort, but could also provide a GPP estimate for
       each combination of DBH and cohort (shape `(4,3)`).

    2. When ``profile=True``, then StemAllocation will only accept a 1D array of GPP
       values but will calculate allocation values for all combinations of DBH, cohort
       and GPP.

    Args:
        cohorts: An instance of :class:`~pyrealm.demography.cohorts.Cohorts`.
        allometry: An instance of :class:`~pyrealm.demography.tmodel.StemAllometry`.
        whole_crown_gpp: An array of GPP values available to a stem at which to model
            allocation (kg C).
        profile: A boolean switch used to calculate profiles of allocation values for
            cohorts at different GPP values.

    TODO::

        Add args to allow inputs from PModel in µgC m2 s-1? Would need to scale to
        growth period though.
    """

    _array_attrs: ClassVar[tuple[str, ...]] = (
        "cohort_id",
        "whole_crown_gpp",
        "sapwood_respiration",
        "foliage_respiration",
        "fine_root_respiration",
        "foliage_turnover",
        "fine_root_turnover",
        "branch_turnover",
        "npp",
    )

    __experimental__ = True

    def __init__(
        self,
        cohorts: Cohorts,
        allometry: StemAllometry,
        whole_crown_gpp: NDArray[np.floating],
        profile: bool = False,
    ) -> None:
        """Calculate allocation of GPP for cohorts."""

        warn_experimental("StemAllocation")

        self.cohort_id: NDArray[np.generic]
        """A numpy array of cohort IDs."""
        self._profile: bool
        """An boolean flag indicating if predictions are for a GPP profile."""
        self._ndims: int
        """An integer giving the dimensionality of the predictions."""

        self.whole_crown_gpp: NDArray[np.floating]
        """An array of gross primary productivity values (kg C) across the whole of the
        crown of each stem to be allocated to respiration, turnover and growth."""
        self.sapwood_respiration: NDArray[np.floating]
        """Allocation to sapwood respiration (g C)"""
        self.foliage_respiration: NDArray[np.floating]
        """Allocation to foliar respiration (g C)"""
        self.fine_root_respiration: NDArray[np.floating]
        """Allocation to fine root respiration (g C)"""
        self.foliage_turnover: NDArray[np.floating]
        """Allocation to leaf turnover (g C)"""
        self.fine_root_turnover: NDArray[np.floating]
        """Allocation to fine root turnover"""
        self.branch_turnover: NDArray[np.floating]
        """Allocation to branch turnover"""
        self.npp: NDArray[np.floating]
        """Net primary productivity (g C)"""

        # Validate GPP input and handle array broadcasting
        self.profile = profile

        # Check we have an array of strictly positive values
        if not isinstance(whole_crown_gpp, np.ndarray):
            raise ValueError("The whole_crown_gpp value must be a numpy array.")
        if np.any(whole_crown_gpp <= 0):
            raise ValueError("Values in whole_crown_gpp must be greater than zero.")

        if self.profile:
            # In profiling mode, a prediction is made at each GPP for each allometry
            # prediction, so broadcast an extra dimensions on to the front of GPP to
            # make it work.
            if whole_crown_gpp.ndim != 1:
                raise ValueError("Allocation profiling requires a 1D array.")
            # Insert new dimensions after the profiling dimension and then broadcast
            self.whole_crown_gpp = whole_crown_gpp[
                :, *[None] * allometry._ndims
            ] * np.ones_like(allometry.dbh)

            self.cohort_id = np.broadcast_to(
                allometry.cohort_id, self.whole_crown_gpp.shape
            )
            self._ndims = allometry._ndims + 1
        else:
            try:
                self.whole_crown_gpp = np.broadcast_to(
                    whole_crown_gpp, allometry.dbh.shape
                )
            except ValueError:
                raise ValueError(
                    f"The GPP array shape ({whole_crown_gpp.shape})is not congruent "
                    f"with predicted allometry shape ({allometry.dbh.shape})."
                )
            self.cohort_id = allometry.cohort_id
            self._ndims = allometry._ndims

        # To handle GPP profiling, the allocation terms that do not rely on GPP -
        # respiration and turnover - need to broadcast their inputs to match.

        # Calculate respiration terms
        self.sapwood_respiration = calculate_sapwood_respiration(
            resp_s=cohorts.resp_s.to_numpy(),
            sapwood_mass=np.broadcast_to(
                allometry.sapwood_mass, self.whole_crown_gpp.shape
            ),
        )

        self.foliage_respiration = calculate_foliage_respiration(
            resp_f=cohorts.resp_f.to_numpy(),
            whole_crown_gpp=self.whole_crown_gpp,
        )

        self.fine_root_respiration = calculate_fine_root_respiration(
            resp_r=cohorts.resp_r.to_numpy(),
            fine_root_mass=np.broadcast_to(
                allometry.fine_root_mass, self.whole_crown_gpp.shape
            ),
        )

        # Calculate NPP given losses to yield fraction and respiration
        self.npp = calculate_net_primary_productivity(
            yld=cohorts.yld.to_numpy(),
            whole_crown_gpp=self.whole_crown_gpp,
            foliage_respiration=self.foliage_respiration,
            fine_root_respiration=self.fine_root_respiration,
            sapwood_respiration=self.sapwood_respiration,
        )

        # Calculate turnover costs
        self.foliage_turnover = calculate_foliage_turnover(
            tau_f=cohorts.tau_f.to_numpy(),
            foliage_mass=np.broadcast_to(
                allometry.foliage_mass, self.whole_crown_gpp.shape
            ),
        )

        self.fine_root_turnover = calculate_fine_root_turnover(
            tau_r=cohorts.tau_r.to_numpy(),
            fine_root_mass=np.broadcast_to(
                allometry.fine_root_mass, self.whole_crown_gpp.shape
            ),
        )

        self.branch_turnover = calculate_branch_turnover(
            tau_b=cohorts.tau_b.to_numpy(),
            stem_mass=np.broadcast_to(allometry.stem_mass, self.whole_crown_gpp.shape),
        )

    def __repr__(self) -> str:
        match (self.profile, self._ndims):
            case (True, 3):
                repr_ = (
                    "StemAllocation: Profiles for {2} cohorts at {1} DBH values"
                    " and {0} GPP values."
                ).format(
                    *self.whole_crown_gpp.shape,
                )
            case (True, 2):
                repr_ = (
                    "StemAllocation: Profiles for {1} cohorts at {0} GPP values."
                ).format(
                    *self.whole_crown_gpp.shape,
                )
            case (False, 2):
                repr_ = (
                    "StemAllocation: Profiles for {1} cohorts at {0} DBH values."
                ).format(
                    *self.whole_crown_gpp.shape,
                )
            case (False, 1):
                repr_ = ("StemAllocation: Values for {} cohorts").format(
                    *self.whole_crown_gpp.shape,
                )

        return repr_


class GrowthIncrements(ToDataFrameMixin):
    """Calculate growth increments using the T Model.

    Stem growth in the T Model is calculated by partitioning biomass production between
    predicted biomass turnover and carbon available for growth. The T model is then used
    to estimate an increase in stem diameter that allocates the carbon available for
    growth across stem, fine roots and foliage according to the allometry of the plant
    functional types.

    This class calculates the stem, fine root and foliage growth increments given:

    * a set of cohorts,
    * the current stem allometry of those cohorts, and
    * calculated allocation of whole crown GPP estimates for the cohorts into
      respiration, NPP and turnover values.

    By default, the calculation will follow the original T Model :cite:`Li:2014bc`, in
    assuming that all of the net primary productivity is available for biomass
    production. However, the ``biomass_production`` argument can be used to use reduce
    NPP to allocate carbon to other pools such as VOC emissions, soil exudates and
    non-structural carbohydrates.

    Args:
        cohorts: A set of cohorts
        allometry: The current stem allometry for those cohorts
        gpp_allocation: An allocation of whole crown GPP for those cohorts.
        biomass_production: An optional array of biomass production values, used to
            override the NPP estimate in ``gpp_allocation``.
    """

    _array_attrs: ClassVar[tuple[str, ...]] = (
        "cohort_id",
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
        stem_allocation: StemAllocation,
        biomass_production: NDArray[np.floating] | None = None,
    ):
        self.cohort_id: NDArray[np.generic]
        """A numpy array of cohort IDs."""
        self._profile: bool
        """An boolean flag indicating if predictions are for a GPP profile."""
        self._ndims: int
        """An integer giving the dimensionality of the predictions."""

        self.biomass_production: NDArray[np.floating]
        """The carbon available for biomass production (g C)"""
        self.delta_dbh: NDArray[np.floating]
        """Predicted increase in stem diameter from growth allocation (m)"""
        self.delta_stem_mass: NDArray[np.floating]
        """Predicted increase in stem mass from growth allocation (g C)"""
        self.delta_foliage_mass: NDArray[np.floating]
        """Predicted increase in foliar mass from growth allocation (g C)"""
        self.delta_fine_root_mass: NDArray[np.floating]
        """Predicted increase in fine root mass from growth allocation (g C)"""

        # Set the biomass production values to be used
        if biomass_production is None:
            biomass_production = stem_allocation.npp
        else:
            if stem_allocation.npp.shape != biomass_production.shape:
                raise ValueError(
                    "The biomass_production has a different shape to the "
                    "stem_allocation predictions."
                )

        self.biomass_production = biomass_production

        # Copy over attributes from stem_allocation
        self.profile = stem_allocation.profile
        self._ndims = stem_allocation._ndims
        self.cohort_id = stem_allocation.cohort_id

        turnover = (
            stem_allocation.fine_root_turnover
            + stem_allocation.foliage_turnover
            + stem_allocation.branch_turnover
        )

        (
            self.delta_dbh,
            self.delta_stem_mass,
            self.delta_foliage_mass,
            self.delta_fine_root_mass,
        ) = calculate_growth_increments(
            rho_s=cohorts.rho_s.to_numpy(),
            a_hd=cohorts.a_hd.to_numpy(),
            h_max=cohorts.h_max.to_numpy(),
            lai=cohorts.lai.to_numpy(),
            ca_ratio=cohorts.ca_ratio.to_numpy(),
            sla=cohorts.sla.to_numpy(),
            zeta=cohorts.zeta.to_numpy(),
            biomass_production=biomass_production,
            turnover=turnover,
            dbh=allometry.dbh,
            stem_height=allometry.stem_height,
        )

    def __repr__(self) -> str:
        match (self.profile, self._ndims):
            case (True, 3):
                repr_ = (
                    "GrowthIncrements: Profiles for {2} cohorts at {1} DBH values"
                    " and {0} GPP values."
                ).format(
                    *self.biomass_production.shape,
                )
            case (True, 2):
                repr_ = (
                    "GrowthIncrements: Profiles for {1} cohorts at {0} GPP values."
                ).format(
                    *self.biomass_production.shape,
                )
            case (False, 2):
                repr_ = (
                    "GrowthIncrements: Profiles for {1} cohorts at {0} DBH values."
                ).format(
                    *self.biomass_production.shape,
                )
            case (False, 1):
                repr_ = ("GrowthIncrements: Values for {} cohorts").format(
                    *self.biomass_production.shape,
                )

        return repr_
