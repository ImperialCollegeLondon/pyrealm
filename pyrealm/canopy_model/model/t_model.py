"""Utilities for calculating mass and geometry using the T model."""

from dataclasses import dataclass

import numpy as np

from pyrealm.canopy_model.model.flora import PlantFunctionalType


@dataclass
class TModelGeometry:
    """Geometry and mass of a cohort, calculated using the T Model.

    Includes the geometry and mass of a particular cohort with a given
    diameter at breast height, calculated using the T Model.
    """

    height: float
    crown_area: float
    crown_fraction: float
    mass_stem: float
    mass_fol: float
    mass_swd: float


def calculate_t_model_geometry(dbh: float, pft: PlantFunctionalType) -> TModelGeometry:
    """Calculate T Model Geometry.

    Calculate the geometry and mass of a plant or cohort with a given diameter at
    breast height and plant functional type.
    """

    height = pft.h_max * (1 - np.exp(-pft.a_hd * dbh / pft.h_max))

    # Crown area of tree, Equation (8) of Li ea.
    crown_area = (np.pi * pft.ca_ratio / (4 * pft.a_hd)) * dbh * height

    # Crown fraction, Equation (11) of Li ea.
    crown_fraction = height / (pft.a_hd * dbh)

    # Masses
    mass_stem = (np.pi / 8) * (dbh**2) * height * pft.rho_s
    mass_fol = crown_area * pft.lai * (1 / pft.sla)
    mass_swd = crown_area * pft.rho_s * height * (1 - crown_fraction / 2) / pft.ca_ratio
    return TModelGeometry(
        height,
        crown_area,
        crown_fraction,
        mass_stem,
        mass_fol,
        mass_swd,
    )
