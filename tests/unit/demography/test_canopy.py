"""Testing the Canopy object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="""
    cohort_args, 
    cohort_fapar, 
    community_lai, 
    community_transmission,
    transmission_to_ground
    """,
    argvalues=(
        [
            pytest.param(
                {
                    "projected_leaf_area": np.tile([[2], [4], [6], [8]], 3),
                    "n_individuals": np.array([2, 2, 2]),
                    "lai": np.array([2, 2, 2]),
                    "par_ext": np.array([0.5, 0.5, 0.5]),
                    "cell_area": 12,
                },
                np.outer(
                    np.cumprod(np.concat([[1], np.repeat(np.exp(-1), 3)])),
                    1 - np.repeat(np.exp(-1), 3),
                ),
                np.repeat(2, 4),
                np.cumprod(np.concat([[1], np.repeat(np.exp(-1), 3)])),
                np.exp(-1) ** 4,
                id="four_layers",
            ),
            pytest.param(
                {
                    "projected_leaf_area": np.cumsum(
                        np.array(
                            [
                                [6, 2, 0],
                                [4, 4, 0],
                                [3, 3, 1],
                                [0, 2, 3],
                                [0, 0, 1],
                            ]
                        ),
                        axis=0,
                    ),
                    "lai": np.array([2, 1, 2]),
                    "par_ext": np.array([0.5, 0.5, 0.6]),
                    "n_individuals": np.array([1, 1, 2]),
                    "cell_area": 8,
                },
                np.array(
                    [
                        [0.632121, 0.393469, 0.698806],
                        [0.270258, 0.168225, 0.298769],
                        [0.131671, 0.08196, 0.145562],
                        [0.058028, 0.03612, 0.064149],
                        [0.021907, 0.013636, 0.024218],
                    ]
                ),
                np.array([1.75, 1.5, 1.625, 1.75, 0.5]),
                np.array([1.0, 0.427542, 0.208301, 0.091799, 0.034657]),
                np.array([0.028602]),
                id="simulation_outputs",
            ),
        ]
    ),
)
def test_CohortCanopyData__init__(
    cohort_args,
    cohort_fapar,
    community_lai,
    community_transmission,
    transmission_to_ground,
):
    """Shared testing of the cohort and community canopy dataclasses.

    Since the creation of the community canopy data is built into the cohort canopy
    data creation, that dataclass is implicitly tested by checking the CohortCanopyData
    class.

    The simple four layer test three identical cohorts to give some easily defined
    expected values. The simulation test data is a simple example used in the light
    capture model documentation notebook. This test includes:

    * varying PFT values in LAI and extinction coefficient
    * differing numbers of individuals and
    * includes an incomplete final layer
    """

    from pyrealm.demography.canopy import CohortCanopyData

    # Calculate canopy components
    instance = CohortCanopyData(**cohort_args)

    # Test cohort expected fapar
    assert_allclose(instance.fapar, cohort_fapar, atol=1e-6)

    # Test community expected lai and transmission
    assert_allclose(
        instance.community_data.transmission_profile, community_transmission, atol=1e-6
    )
    assert_allclose(instance.community_data.average_layer_lai, community_lai, atol=1e-6)

    # Test the inherited to_pandas method of the cohort canopy data
    df = instance.to_pandas()

    assert df.shape == (
        np.prod(cohort_args["projected_leaf_area"].shape),
        len(instance.array_attrs),
    )

    assert set(instance.array_attrs) == set(df.columns)

    # Test the inherited to_pandas method of the community canopy data
    df = instance.community_data.to_pandas()

    assert df.shape == (
        len(instance.community_data.transmission_profile),
        len(instance.community_data.array_attrs),
    )

    assert set(instance.community_data.array_attrs) == set(df.columns)


def test_Canopy__init__():
    """Test happy path for initialisation.

    test that when a new canopy object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.canopy import Canopy
    from pyrealm.demography.community import Cohorts, Community
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
        cohorts=Cohorts(
            pft_names=np.array(["broadleaf", "conifer"]),
            n_individuals=np.array([6, 1]),
            dbh_values=np.array([0.2, 0.5]),
        ),
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
    assert canopy.cohort_data.stem_leaf_area.shape == (
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
        np.flip(fixture_community.stem_allometry.crown_z_max.flatten()),
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
