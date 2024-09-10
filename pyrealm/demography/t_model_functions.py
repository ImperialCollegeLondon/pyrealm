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
    of the plant functional type :cite:p:`Li:2014bc`:

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
    functional type :cite:p:`{Equation 14, }Li:2014bc`:

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
    """Calculate z_m, the height of maximum crown radius."""

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
