"""The ``t_model`` module provides the basic scaling relationships of the T Model
:cite:`Li:2014bc`:. This provides scaling relationships using the plant functional type
traits defined in the :mod:`~pyrealm.demography.flora` module and the diameter at breast
height of individual stems to define the stem geometry, masses, respiration and hence
calculate stem growth given net primary productivity.
"""  # noqa: D205

from dataclasses import InitVar, dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.utilities import check_input_shapes
from pyrealm.demography.canopy_functions import (
    calculate_crown_r0,
    calculate_crown_z_max,
)
from pyrealm.demography.flora import Flora, StemTraits


def _validate_t_model_args(pft_args: list[NDArray], size_args: list[NDArray]) -> None:
    """Shared validation for T model function inputs.

    Args:
        pft_args: A list of row arrays representing trait values
        size_args: A list of arrays representing points in the stem size and growth
            allometries at which to evaluate functions.
    """

    # Check PFT inputs all line up and are 1D (row) arrays
    try:
        pft_args_shape = check_input_shapes(*pft_args)
    except ValueError:
        raise ValueError("PFT trait values are not of equal length")

    if len(pft_args_shape) > 1:
        raise ValueError("T model functions only accept 1D arrays of PFT trait values")

    # Check size and growth inputs
    try:
        size_args_shape = check_input_shapes(*size_args)
    except ValueError:
        raise ValueError("Size arrays are not of equal length")

    # Explicitly check to see if the size arrays are row arrays and - if so - enforce
    # that they are the same length.abs

    if len(size_args_shape) == 1 and not pft_args_shape == size_args_shape:
        raise ValueError("Trait and size inputs are row arrays of unequal length.")

    # Otherwise use np.broadcast_shapes to catch issues
    try:
        _ = np.broadcast_shapes(pft_args_shape, size_args_shape)
    except ValueError:
        raise ValueError("PFT and size inputs to T model function are not compatible.")


def calculate_heights(
    h_max: NDArray[np.float32],
    a_hd: NDArray[np.float32],
    dbh: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
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
        _validate_t_model_args(pft_args=[h_max, a_hd], size_args=[dbh])

    return h_max * (1 - np.exp(-a_hd * dbh / h_max))


def calculate_dbh_from_height(
    h_max: NDArray[np.float32],
    a_hd: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    r"""Calculate diameter at breast height from stem height under the T Model.

    This function inverts the normal calculation of stem height (:math:`H`) from
    diameter at breast height (DBH, :math:`D`) in the T Model (see
    :meth:`~pyrealm.demography.t_model_functions.calculate_heights`). This is a helper
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
        _validate_t_model_args(pft_args=[h_max, a_hd], size_args=[stem_height])

    # The equation here blows up in a couple of ways:
    # - H > h_max leads to negative logs which generates np.nan with an invalid value
    #   warning. The np.nan here is what we want to happen, so the warning needs
    #   suppressing.
    # - H = h_max generates a divide by zero which returns inf with a warning. Here the
    #   answer should be h_max so that needs trapping.

    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "divide by zero encountered in divide")
        sup.filter(RuntimeWarning, "invalid value encountered in log")
        return (h_max * np.log(h_max / (h_max - stem_height))) / a_hd


def calculate_crown_areas(
    ca_ratio: NDArray[np.float32],
    a_hd: NDArray[np.float32],
    dbh: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
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
        stem_height: Stem height of individuals
        validate: Boolean flag to suppress argument validation
    """

    if validate:
        _validate_t_model_args(pft_args=[ca_ratio, a_hd], size_args=[dbh, stem_height])

    return ((np.pi * ca_ratio) / (4 * a_hd)) * dbh * stem_height


def calculate_crown_fractions(
    a_hd: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    dbh: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    r"""Calculate tree crown fraction under the T Model.

    The crown fraction (:math:`f_{c}`)is calculated from individual diameters at breast
    height and stem height (:math:`D`), along with the initial slope of the height /
    diameter relationship (:math:`a`) of the plant functional type
    :cite:p:`{Equation 11, }Li:2014bc`:

    .. math::

        \frac{H}{a D}

    Args:
        a_hd: Initial slope of the height/diameter relationship of the PFT
        stem_height: Stem height of individuals
        dbh: Diameter at breast height of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        _validate_t_model_args(pft_args=[a_hd], size_args=[dbh, stem_height])

    return stem_height / (a_hd * dbh)


def calculate_stem_masses(
    rho_s: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    dbh: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    r"""Calculate stem mass under the T Model.

    The stem mass (:math:`W_{s}`) is calculated from individual diameters at breast
    height (:math:`D`) and stem height (:math:`H`), along with the wood density
    (:math:`\rho_s`)of the plant functional type :cite:p:`{Equation 6, }Li:2014bc`:

    .. math::

        W_s = (\pi / 8) \rho_s D^2 H

    Args:
        rho_s: Wood density of the PFT
        stem_height: Stem height of individuals
        dbh: Diameter at breast height of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        _validate_t_model_args(pft_args=[rho_s], size_args=[dbh, stem_height])

    return (np.pi / 8) * rho_s * (dbh**2) * stem_height


def calculate_foliage_masses(
    sla: NDArray[np.float32],
    lai: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
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
        _validate_t_model_args(pft_args=[sla, lai], size_args=[crown_area])

    return crown_area * lai * (1 / sla)


def calculate_sapwood_masses(
    rho_s: NDArray[np.float32],
    ca_ratio: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    crown_fraction: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
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
        stem_height: Stem height of individuals
        crown_area: Crown area of individuals
        crown_fraction: Crown fraction of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        _validate_t_model_args(
            pft_args=[rho_s, ca_ratio],
            size_args=[stem_height, crown_area, crown_fraction],
        )

    return crown_area * rho_s * stem_height * (1 - crown_fraction / 2) / ca_ratio


def calculate_whole_crown_gpp(
    potential_gpp: NDArray[np.float32],
    crown_area: NDArray[np.float32],
    par_ext: NDArray[np.float32],
    lai: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    r"""Calculate whole crown gross primary productivity.

    This function calculates individual GPP across the whole crown, given the individual
    potential gross primary productivity (GPP) per metre squared (:math:`P_0`) and crown
    area (:math:`A_c`), along with the leaf area index (:math:`L`) and the extinction
    coefficient (:math:`k`) of the plant functional type :cite:p:`{Equation 12,
    }Li:2014bc`.

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
        _validate_t_model_args(
            pft_args=[lai, par_ext], size_args=[potential_gpp, crown_area]
        )

    return potential_gpp * crown_area * (1 - np.exp(-(par_ext * lai)))


def calculate_sapwood_respiration(
    resp_s: NDArray[np.float32],
    sapwood_mass: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
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
        _validate_t_model_args(pft_args=[resp_s], size_args=[sapwood_mass])

    return sapwood_mass * resp_s


def calculate_foliar_respiration(
    resp_f: NDArray[np.float32],
    whole_crown_gpp: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
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
        _validate_t_model_args(pft_args=[resp_f], size_args=[whole_crown_gpp])

    return whole_crown_gpp * resp_f


def calculate_fine_root_respiration(
    zeta: NDArray[np.float32],
    sla: NDArray[np.float32],
    resp_r: NDArray[np.float32],
    foliage_mass: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    r"""Calculate foliar respiration.

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
        _validate_t_model_args(pft_args=[zeta, sla, resp_r], size_args=[foliage_mass])

    return zeta * sla * foliage_mass * resp_r


def calculate_net_primary_productivity(
    yld: NDArray[np.float32],
    whole_crown_gpp: NDArray[np.float32],
    foliar_respiration: NDArray[np.float32],
    fine_root_respiration: NDArray[np.float32],
    sapwood_respiration: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
    r"""Calculate net primary productivity.

    The net primary productivity (NPP, :math:`P_{net}`) is calculated as a plant
    functional type specific yield proportion (:math:`y`) of the total GPP (:math:`P`)
    for the individual minus respiration (:math:`R_m`), as the sum of the respiration
    costs for foliage  (:math:`R_f`), fine roots  (:math:`R_r`) and sapwood
    (:math:`R_s`).

    .. math::
        P_{net} = y (P - R_m) = y (P - W_{\cdot s} r_s - \zeta \sigma W_f r_r - P r_f)

    Note that this differs from Equation 13 of :cite:t:`Li:2014bc`, which does not
    include a term for foliar respiration. This is because :cite:t:`Li:2014bc` remove
    foliar respiration as a fixed proportion of potential GPP as the first step in their
    calculations. The approach here is equivalent but allows the foliar respiration to
    vary between plant functional types.

    Args:
        yld: The yield proportion.
        whole_crown_gpp: The total GPP for the crown.
        foliar_respiration: The total foliar respiration.
        fine_root_respiration: The total fine root respiration
        sapwood_respiration: The total sapwood respiration.
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        _validate_t_model_args(
            pft_args=[yld],
            size_args=[
                whole_crown_gpp,
                foliar_respiration,
                fine_root_respiration,
                sapwood_respiration,
            ],
        )

    return yld * (
        whole_crown_gpp
        - foliar_respiration
        - fine_root_respiration
        - sapwood_respiration
    )


def calculate_foliage_and_fine_root_turnover(
    sla: NDArray[np.float32],
    zeta: NDArray[np.float32],
    tau_f: NDArray[np.float32],
    tau_r: NDArray[np.float32],
    foliage_mass: NDArray[np.float32],
    validate: bool = True,
) -> NDArray[np.float32]:
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
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        _validate_t_model_args(
            pft_args=[sla, zeta, tau_f, tau_r], size_args=[foliage_mass]
        )

    return foliage_mass * ((1 / tau_f) + (sla * zeta / tau_r))


def calculate_growth_increments(
    rho_s: NDArray[np.float32],
    a_hd: NDArray[np.float32],
    h_max: NDArray[np.float32],
    lai: NDArray[np.float32],
    ca_ratio: NDArray[np.float32],
    sla: NDArray[np.float32],
    zeta: NDArray[np.float32],
    npp: NDArray[np.float32],
    turnover: NDArray[np.float32],
    dbh: NDArray[np.float32],
    stem_height: NDArray[np.float32],
    validate: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    r"""Calculate growth increments.

    Given an estimate of net primary productivity (:math:`P_{net}`), less associated  
    turnover costs (:math:`T`), the remaining productivity can be allocated to growth
    and hence estimate resulting increments :cite:`Li:2014bc` in:
    
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
        dbh: Diameter at breast height of individuals
        stem_height: Stem height of individuals
        validate: Boolean flag to suppress argument validation
    """
    if validate:
        _validate_t_model_args(
            pft_args=[rho_s, a_hd, h_max, lai, ca_ratio, sla, zeta],
            size_args=[npp, turnover, dbh, stem_height],
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
        * (1 / sla + zeta)
    )

    # Increment of diameter at breast height
    delta_d = (npp - turnover) / (dWsdt + dWfdt)

    return (delta_d, dWsdt * delta_d, dWfdt * delta_d)


@dataclass
class StemAllometry:
    """Calculate T Model allometric predictions across a set of stems.

    This method calculate predictions of stem allometries for stem height, crown area,
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

    allometry_attrs: ClassVar[tuple[str, ...]] = (
        "dbh",
        "stem_height",
        "crown_area",
        "crown_fraction",
        "stem_mass",
        "foliage_mass",
        "sapwood_mass",
        "crown_r0",
        "crown_z_max",
    )

    # Init vars
    stem_traits: InitVar[Flora | StemTraits]
    """ An instance of :class:`~pyrealm.demography.flora.Flora` or 
    :class:`~pyrealm.demography.flora.StemTraits`, providing plant functional trait data
    for a set of stems."""
    at_dbh: InitVar[NDArray[np.float32]]
    """An array of diameter at breast height values at which to predict stem allometry 
    values."""

    # Post init allometry attributes
    dbh: NDArray[np.float32] = field(init=False)
    """Diameter at breast height (metres)"""
    stem_height: NDArray[np.float32] = field(init=False)
    """Stem height (metres)"""
    crown_area: NDArray[np.float32] = field(init=False)
    """Crown area (square metres)"""
    crown_fraction: NDArray[np.float32] = field(init=False)
    """Vertical fraction of the stem covered by the crown (-)"""
    stem_mass: NDArray[np.float32] = field(init=False)
    """Stem mass (kg)"""
    foliage_mass: NDArray[np.float32] = field(init=False)
    """Foliage mass (kg)"""
    sapwood_mass: NDArray[np.float32] = field(init=False)
    """Sapwood mass (kg)"""
    crown_r0: NDArray[np.float32] = field(init=False)
    """Crown radius scaling factor (-)"""
    crown_z_max: NDArray[np.float32] = field(init=False)
    """Height of maximum crown radius (metres)"""

    def __post_init__(
        self, stem_traits: Flora | StemTraits, at_dbh: NDArray[np.float32]
    ) -> None:
        """Populate the stem allometry attributes from the traits and size data."""

        self.stem_height = calculate_heights(
            h_max=stem_traits.h_max,
            a_hd=stem_traits.a_hd,
            dbh=at_dbh,
        )

        # Broadcast at_dbh to shape of stem height to get congruent shapes
        self.dbh = np.broadcast_to(at_dbh, self.stem_height.shape)

        self.crown_area = calculate_crown_areas(
            ca_ratio=stem_traits.ca_ratio,
            a_hd=stem_traits.a_hd,
            dbh=self.dbh,
            stem_height=self.stem_height,
        )

        self.crown_fraction = calculate_crown_fractions(
            a_hd=stem_traits.a_hd,
            dbh=self.dbh,
            stem_height=self.stem_height,
        )

        self.stem_mass = calculate_stem_masses(
            rho_s=stem_traits.rho_s,
            dbh=self.dbh,
            stem_height=self.stem_height,
        )

        self.foliage_mass = calculate_foliage_masses(
            sla=stem_traits.sla,
            lai=stem_traits.lai,
            crown_area=self.crown_area,
        )

        self.sapwood_mass = calculate_sapwood_masses(
            rho_s=stem_traits.rho_s,
            ca_ratio=stem_traits.ca_ratio,
            stem_height=self.stem_height,
            crown_area=self.crown_area,
            crown_fraction=self.crown_fraction,
        )

        self.crown_r0 = calculate_crown_r0(
            q_m=stem_traits.q_m,
            crown_area=self.crown_area,
        )

        self.crown_z_max = calculate_crown_z_max(
            z_max_prop=stem_traits.z_max_prop,
            stem_height=self.stem_height,
        )


@dataclass()
class StemAllocation:
    """Calculate T Model allocation predictions across a set of stems.

    This method calculate predictions of allocation of potential GPP for stems under the
    T Model :cite:`Li:2014bc`, given a set of traits for those stems and the stem
    allometries given the stem size.

    Args:
        stem_traits: An instance of :class:`~pyrealm.demography.flora.Flora` or
            :class:`~pyrealm.demography.flora.StemTraits`, providing plant functional
            trait data for a set of stems.
        stem_allometry: An instance of
            :class:`~pyrealm.demography.t_model_functions.StemAllometry`
            providing the stem size data for which to calculate allocation.
        at_potential_gpp: An array of diameter at breast height values at which to
            predict stem allometry values.
    """

    allocation_attrs: ClassVar[tuple[str, ...]] = (
        "potential_gpp",
        "whole_crown_gpp",
        "sapwood_respiration",
        "foliar_respiration",
        "fine_root_respiration",
        "npp",
        "turnover",
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
    """An instance of :class:`~pyrealm.demography.t_model_functions.StemAllometry`
    providing the stem size data for which to calculate allocation."""
    at_potential_gpp: InitVar[NDArray[np.float32]]
    """An array of potential gross primary productivity for each stem that should be
    allocated to respiration, turnover and growth."""

    # Post init allometry attributes
    potential_gpp: NDArray[np.float32] = field(init=False)
    """Potential GPP per unit area (g C m2)"""
    whole_crown_gpp: NDArray[np.float32] = field(init=False)
    """Estimated GPP across the whole crown (g C)"""
    sapwood_respiration: NDArray[np.float32] = field(init=False)
    """Allocation to sapwood respiration (g C)"""
    foliar_respiration: NDArray[np.float32] = field(init=False)
    """Allocation to foliar respiration (g C)"""
    fine_root_respiration: NDArray[np.float32] = field(init=False)
    """Allocation to fine root respiration (g C)"""
    npp: NDArray[np.float32] = field(init=False)
    """Net primary productivity (g C)"""
    turnover: NDArray[np.float32] = field(init=False)
    """Allocation to leaf and fine root turnover (g C)"""
    delta_dbh: NDArray[np.float32] = field(init=False)
    """Predicted increase in stem diameter from growth allocation (g C)"""
    delta_stem_mass: NDArray[np.float32] = field(init=False)
    """Predicted increase in stem mass from growth allocation (g C)"""
    delta_foliage_mass: NDArray[np.float32] = field(init=False)
    """Predicted increase in foliar mass from growth allocation (g C)"""

    def __post_init__(
        self,
        stem_traits: Flora | StemTraits,
        stem_allometry: StemAllometry,
        at_potential_gpp: NDArray[np.float32],
    ) -> None:
        """Populate stem allocation attributes from the traits, allometry and GPP."""

        # Broadcast potential GPP to match stem data outputs
        self.potential_gpp = np.broadcast_to(at_potential_gpp, stem_allometry.dbh.shape)

        self.whole_crown_gpp = calculate_whole_crown_gpp(
            potential_gpp=self.potential_gpp,
            crown_area=stem_allometry.crown_area,
            par_ext=stem_traits.par_ext,
            lai=stem_traits.lai,
        )

        self.sapwood_respiration = calculate_sapwood_respiration(
            resp_s=stem_traits.resp_s, sapwood_mass=stem_allometry.sapwood_mass
        )

        self.foliar_respiration = calculate_foliar_respiration(
            resp_f=stem_traits.resp_f, whole_crown_gpp=self.whole_crown_gpp
        )

        self.fine_root_respiration = calculate_fine_root_respiration(
            zeta=stem_traits.zeta,
            sla=stem_traits.sla,
            resp_r=stem_traits.resp_r,
            foliage_mass=stem_allometry.foliage_mass,
        )

        self.npp = calculate_net_primary_productivity(
            yld=stem_traits.yld,
            whole_crown_gpp=self.whole_crown_gpp,
            foliar_respiration=self.foliar_respiration,
            fine_root_respiration=self.fine_root_respiration,
            sapwood_respiration=self.sapwood_respiration,
        )

        self.turnover = calculate_foliage_and_fine_root_turnover(
            sla=stem_traits.sla,
            zeta=stem_traits.zeta,
            tau_f=stem_traits.tau_f,
            tau_r=stem_traits.tau_r,
            foliage_mass=stem_allometry.foliage_mass,
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
                turnover=self.turnover,
                dbh=stem_allometry.dbh,
                stem_height=stem_allometry.stem_height,
            )
        )
