"""Testing the Canopy object."""

import numpy as np
import pytest


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
    canopy = Canopy(community, canopy_gap_fraction=canopy_gap_fraction, fit_ppa=True)

    # Simply check that the shape of the stem leaf area matrix is the right shape
    n_layers_from_crown_area = int(
        np.ceil(
            (
                (
                    community.stem_allometry.crown_area
                    * community.cohorts.n_individuals
                ).sum()
                * (1 + canopy_gap_fraction)
            )
            / community.cell_area
        )
    )
    assert canopy.stem_leaf_area.shape == (
        n_layers_from_crown_area,
        canopy.n_cohorts,
    )


def test_solve_canopy_area_filling_height(fixture_community):
    """Test solve_community_projected_canopy_area.

    The logic of this test is that given the cumulative sum of the crown areas in the
    fixture from tallest to shortest as the target, providing the z_max of each stem as
    the height _should_ always return zero, as this is exactly the height at which that
    cumulative area would close: crown 1 closes at z_max 1, crown 1 + 2 closes at z_max
    2 and so on.
    """

    from pyrealm.demography.canopy import (
        solve_canopy_area_filling_height,
    )

    for (
        this_height,
        this_target,
    ) in zip(
        np.flip(fixture_community.stem_allometry.crown_z_max),
        np.cumsum(np.flip(fixture_community.stem_allometry.crown_area)),
    ):
        solved = solve_canopy_area_filling_height(
            z=this_height,
            stem_height=fixture_community.stem_allometry.stem_height,
            crown_area=fixture_community.stem_allometry.crown_area,
            n_individuals=fixture_community.cohorts.n_individuals,
            m=fixture_community.stem_traits.m,
            n=fixture_community.stem_traits.n,
            q_m=fixture_community.stem_traits.q_m,
            z_max=fixture_community.stem_allometry.crown_z_max,
            target_area=this_target,
        )

    assert solved == pytest.approx(0)
