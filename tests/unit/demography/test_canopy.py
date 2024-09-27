"""Testing the Canopy object."""

import numpy as np


def test_Canopy__init__():
    """Test happy path for initialisation.

    test that when a new canopy object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.canopy import Canopy
    from pyrealm.demography.community import Community
    from pyrealm.demography.flora import Flora, PlantFunctionalType

    flora = Flora(
        [
            PlantFunctionalType(name="broadleaf", h_max=30),
            PlantFunctionalType(name="conifer", h_max=20),
        ]
    )

    community = Community(
        cell_id=1,
        cell_area=20,
        cohort_pft_names=np.array(["broadleaf", "conifer"]),
        cohort_n_individuals=np.array([6, 1]),
        cohort_dbh_values=np.array([0.2, 0.5]),
        flora=flora,
    )

    canopy_gap_fraction = 0.05
    canopy = Canopy(community, canopy_gap_fraction=canopy_gap_fraction)

    # Simply check that the shape of the stem leaf area matrix is the right shape
    n_layers_from_crown_area = int(
        np.ceil(
            (
                (
                    community.stem_allometry.crown_area
                    * community.cohort_data["n_individuals"]
                ).sum()
                * (1 + canopy_gap_fraction)
            )
            / community.cell_area
        )
    )
    assert canopy.stem_leaf_area.shape == (n_layers_from_crown_area, canopy.n_cohorts)
