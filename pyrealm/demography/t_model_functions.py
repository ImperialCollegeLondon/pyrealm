"""The ``t_model`` module provides the basic scaling relationships of the T Model
:cite:`Li:2014bc`:. This provides scaling relationships using the plant functional type
traits defined in the :mod:`~pyrealm.demography.flora` module and the diameter at breast
height of individual stems to define the stem geometry, masses, respiration and hence
the calculate stem growth given net primary productivity.
"""  # noqa: D205

import numpy as np
from pandas import Series


def calculate_heights(h_max: Series, a_hd: Series, dbh: Series) -> Series:
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


def calculate_crown_areas(
    ca_ratio: Series, a_hd: Series, dbh: Series, height: Series
) -> Series:
    r"""Calculate tree crown area under the T Model.

    The tree crown area (:math:`A_{c}`)is calculated from individual diameters at breast
    height (:math:`D`) and stem height (:math:`H`), along with the crown area ratio
    (:math:`c`)and the initial slope of the height/diameter relationship (:math:`a`) of
    the plant functional type :cite:p:`{Equation 8, }Li:2014bc`:

    .. math::

        A_{c} =\frac{\pi c}{4 a} D H


    Args:
        ca_ratio: Crown area ratio of the PFT
        a_hd: Initial slope of the height/diameter relationship of the PFT
        dbh: Diameter at breast height of individuals
        height: Stem height of individuals
    """

    return ((np.pi * ca_ratio) / (4 * a_hd)) * dbh * height


def calculate_crown_fractions(a_hd: Series, height: Series, dbh: Series) -> Series:
    r"""Calculate tree crown fraction under the T Model.

    The crown fraction (:math:`f_{c}`)is calculated from individual diameters at breast
    height and stem height (:math:`D`), along with the initial slope of the height /
    diameter relationship (:math:`a`) of the plant functional type
    :cite:p:`{Equation 11, }Li:2014bc`:

    .. math::

        \frac{H}{a D}

    Args:
        a_hd: Initial slope of the height/diameter relationship of the PFT
        dbh: Diameter at breast height of individuals
        height: Stem height of individuals
    """

    return height / (a_hd * dbh)


def calculate_stem_masses(rho_s: Series, height: Series, dbh: Series) -> Series:
    r"""Calculate stem mass under the T Model.

    The stem mass (:math:`W_{s}`) is calculated from individual diameters at breast
    height (:math:`D`) and stem height (:math:`H`), along with the wood density
    (:math:`\rho_s`)of the plant functional type :cite:p:`{Equation 6, }Li:2014bc`:

    .. math::

        W_s = (\pi / 8) \rho_s D^2 H

    Args:
        rho_s: Wood density of the PFT
        dbh: Diameter at breast height of individuals
        height: Stem height of individuals
    """

    return (np.pi / 8) * rho_s * (dbh**2) * height


def calculate_foliage_masses(sla: Series, lai: Series, crown_area: Series) -> Series:
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


def calculate_sapwood_masses(
    rho_s: Series,
    ca_ratio: Series,
    height: Series,
    crown_area: Series,
    crown_fraction: Series,
) -> Series:
    r"""Calculate sapwood mass under the T Model.

    The sapwood mass (:math:`W_{\cdot s}`) is calculated from the individual crown area
    (:math:`A_{c}`), height :math:`H` and canopy fraction (:math:`f_{c}`) along with the
    wood density (:math:`\rho_s`) and crown area ratio :math:`A_{c}` of the  plant
    functional type :cite:p:`{Equation 14, }Li:2014bc`.

    .. math::

        W_{\cdot s} = \frac{A_c \rho_s H (1 - f_c / 2)}{c}

    Args:
        rho_s: Wood density of the PFT
        ca_ratio: Crown area ratio of the PFT
        height: Stem height of individuals
        crown_area: Crown area of individuals
        crown_fraction: Crown fraction of individuals
    """

    return crown_area * rho_s * height * (1 - crown_fraction / 2) / ca_ratio


def calculate_whole_crown_gpp(
    potential_gpp: Series, crown_area: Series, par_ext: Series, lai: Series
) -> Series:
    r"""Calculate whole crown gross primary productivity.

    This function calculates individual GPP across the whole crown, given  the
    individual potential gross primary productivity (GPP) per metre squared
    (:math:`P_0`) and crown area (:math:`A_c`), along with the leaf area index
    (:math:`L`) and the extinction coefficient (:math:`k`) of the plant functional type
    :cite:p:`{Equation 12, }Li:2014bc`.

    .. math::

        P = P_0 A_c (1 - e^{-kL})

    Args:
        potential_gpp: Potential GPP per metre squared
        crown_area: The crown area in metres squared
        par_ext: The extinction coefficient
        lai: The leaf area index
    """

    return potential_gpp * crown_area * (1 - np.exp(-(par_ext * lai)))


def calculate_sapwood_respiration(resp_s: Series, sapwood_mass: Series) -> Series:
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


def calculate_foliar_respiration(resp_f: Series, whole_crown_gpp: Series) -> Series:
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
    zeta: Series, sla: Series, resp_r: Series, foliage_mass: Series
) -> Series:
    r"""Calculate foliar respiration.

    Calculates the total fine root respiration (:math:`R_{r}`) given the individual
    foliage mass (:math:`W_f`), along with the fine root respiration rate (:math:`r_r`),
    the ratio of fine root mass to foliage area (:math:`\zeta`) and the specific leaf
    area (:math:`\sigma`) :cite:p:`{see Equation 13, }Li:2014bc`

    .. math::
         R_{r} = \zeta \sigma W_f r_r

    Args:
        zeta: The ratio of fine root mass to foliage area.
        sla: The specific leaf area
        resp_r: The respiration rate of fine roots.
        foliage_mass: The individual foliage mass.
    """

    return zeta * sla * foliage_mass * resp_r


def calculate_net_primary_productivity(
    yld: Series,
    whole_crown_gpp: Series,
    foliar_respiration: Series,
    fine_root_respiration: Series,
    sapwood_respiration: Series,
) -> Series:
    r"""Calculate net primary productivity.

    The net primary productivity (NPP, :math:`P_{net}`) is calculated as a plant
    functional type specific yield proportion (:math:`y`) of the total GPP (:math:`P`)
    for the individual minus respiration (:math:`R_m`), as the sum of the respiration
    costs for foliage  (:math:`R_f`), fine roots  (:math:`R_r`) and sapwood
    (:math:`R_s`).

    .. math::
        P_{net} = y (P - R_m) = y (P - W_{\cdot s} r_s - \zeta \sigma W_f r_r - P r_f)

    Note that this differs from Equation 13 of :cite:t:`Li:2014bc`, which removes foliar
    respiration from potential GPP. This approach is equivalent but allows the foliar
    respiration to vary between plant functional types.

    Args:
        yld: The yield proportion.
        whole_crown_gpp: The total GPP for the crown.
        foliar_respiration: The total foliar respiration.
        fine_root_respiration: The total fine root respiration
        sapwood_respiration: The total sapwood respiration.
    """

    return yld * (
        whole_crown_gpp
        - foliar_respiration
        - fine_root_respiration
        - sapwood_respiration
    )


def calculate_foliage_and_fine_root_turnover(
    sla: Series,
    zeta: Series,
    tau_f: Series,
    tau_r: Series,
    foliage_mass: Series,
) -> Series:
    r"""Calculate turnover costs.

    This function calculates the costs associated with the turnover of fine roots and
    foliage. This is calculated from the total foliage mass of individuals
    (:math:`W_f`), along with the specific leaf area (:math:`\sigma`) and fine root mass
    to foliar area ratio (:math:`\zeta`) and the turnover times of foliage
    (:math:`\tau_f`) and fine roots (:math:`\tau_r`) of the plant functional type
    :cite:p:`{see Equation 15, }Li:2014bc`.

    .. math::

        T = W_f \left( \frac{1}{\tau_f} + \frac{\sigma \zeta}{\tau_f} \right)

    Args:
        sla: The specific leaf area
        zeta: The ratio of fine root mass to foliage area.
        tau_f: The turnover time of foliage
        tau_r: The turnover time of fine roots
        foliage_mass: The foliage mass
    """

    return foliage_mass * ((1 / tau_f) + (sla * zeta / tau_r))


def calculate_growth_increments(
    rho_s: Series,
    a_hd: Series,
    h_max: Series,
    lai: Series,
    ca_ratio: Series,
    sla: Series,
    zeta: Series,
    npp: Series,
    turnover: Series,
    dbh: Series,
    height: Series,
) -> tuple[Series, Series, Series]:
    r"""Calculate growth increments.

    Given an estimate of net primary productivity (:math:`P_{net}`), less associated  
    turnover costs (:math:`T`), the remaining productivity can be allocated to growth
    and hence estimate resulting increments in:
    
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

    """
    # Rates of change in stem and foliar
    dWsdt = np.pi / 8 * rho_s * dbh * (a_hd * dbh * (1 - (height / h_max)) + 2 * height)

    dWfdt = (
        lai
        * ((np.pi * ca_ratio) / (4 * a_hd))
        * (a_hd * dbh * (1 - height / h_max) + height)
        * (1 / sla + zeta)
    )

    # Increment of diameter at breast height
    delta_d = (npp - turnover) / (dWsdt + dWfdt)

    return (delta_d, dWsdt * delta_d, dWfdt * delta_d)


def calculate_canopy_q_m(m: float, n: float) -> float:
    """Calculate a q_m value.

    The value of q_m is a constant canopy scaling parameter derived from the ``m`` and
    ``n`` attributes defined for a plant functional type.

    Args:
        m: Canopy shape parameter
        n: Canopy shape parameter
    """
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_canopy_z_max_proportion(m: float, n: float) -> float:
    r"""Calculate the z_m proportion.

    The z_m proportion (:math:`p_{zm}`) is the constant proportion of stem height at
    which the maximum crown radius is found for a given plant functional type.

    .. math::

        p_{zm} = \left(\dfrac{n-1}{m n -1}\right)^ {\tfrac{1}{n}}

    Args:
        m: Canopy shape parameter
        n: Canopy shape parameter
    """

    return ((n - 1) / (m * n - 1)) ** (1 / n)


def calculate_canopy_z_max(z_max_prop: Series, height: Series) -> Series:
    r"""Calculate height of maximum crown radius.

    The height of the maximum crown radius (:math:`z_m`) is derived from the canopy
    shape parameters (:math:`m,n`) and the resulting fixed proportion (:math:`p_{zm}`)
    for plant functional types. These shape parameters are defined as part of the
    extension of the T Model presented by :cite:t:`joshi:2022a`.

    The value :math:`z_m` is the height above ground where the largest canopy radius is
    found, given the proportion and the estimated stem height (:math:`H`) of
    individuals.

    .. math::

        z_m = p_{zm} H

    Args:
        z_max_prop: Canopy shape parameter of the PFT
        height: Crown area of individuals
    """

    return height * z_max_prop


def calculate_canopy_r0(q_m: Series, crown_area: Series) -> Series:
    r"""Calculate scaling factor for height of maximum crown radius.

    This scaling factor (:math:`r_0`) is derived from the canopy shape parameters
    (:math:`m,n,q_m`) for plant functional types and the estimated crown area
    (:math:`A_c`) of individuals. The shape parameters are defined as part of the
    extension of the T Model presented by :cite:t:`joshi:2022a` and :math:`r_0` is used
    to scale the crown area such that the crown area at the  maximum crown radius fits
    the expectations of the T Model.

    .. math::

        r_0 = 1/q_m  \sqrt{A_c / \pi}

    Args:
        q_m: Canopy shape parameter of the PFT
        crown_area: Crown area of individuals
    """
    # Scaling factor to give expected A_c (crown area) at
    # z_m (height of maximum crown radius)

    return 1 / q_m * np.sqrt(crown_area / np.pi)


def calculate_relative_canopy_radii(
    z: float,
    height: Series,
    m: Series,
    n: Series,
) -> Series:
    r"""Calculate relative canopy radius at a given height.

    The canopy shape parameters ``m`` and ``n`` define the vertical distribution of
    canopy along the stem. For a stem of a given total height, this function calculates
    the relative canopy radius at a given height :math:`z`:

    .. math::

        q(z) = m n \left(\dfrac{z}{H}\right) ^ {n -1}
        \left( 1 - \left(\dfrac{z}{H}\right) ^ n \right)^{m-1}

    Args:
        z: Height at which to calculate relative radius
        height: Total height of individual stem
        m: Canopy shape parameter of PFT
        n: Canopy shape parameter of PFT
    """

    z_over_height = z / height

    return m * n * z_over_height ** (n - 1) * (1 - z_over_height**n) ** (m - 1)
