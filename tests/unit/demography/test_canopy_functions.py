"""test the functions in canopy_functions.py."""

import numpy as np
import pytest


@pytest.fixture
def fixture_canopy_shape():
    """Fixture providing input and expected values for shape parameter calculations.

    These are hand calculated and only really test that the calculations haven't changed
    from the initial implementation.
    """
    return {
        "m": np.array([2, 3]),
        "n": np.array([5, 4]),
        "q_m": np.array([2.9038988210485766, 2.3953681843215673]),
        "p_zm": np.array([0.850283, 0.72265688]),
    }


@pytest.fixture
def fixture_community():
    """A fixture providing a simple community."""
    from pyrealm.demography.community import Community
    from pyrealm.demography.flora import Flora, PlantFunctionalType

    # A simple community containing one sample stem
    flora = Flora([PlantFunctionalType(name="test")])
    return Community(
        cell_id=1,
        cell_area=100,
        flora=flora,
        cohort_n_individuals=np.repeat([1], 4),
        cohort_pft_names=np.repeat(["test"], 4),
        cohort_dbh_values=np.array([0.2, 0.4, 0.6, 0.8]),
    )


def test_calculate_canopy_q_m(fixture_canopy_shape):
    """Test calculate_canopy_q_m."""

    from pyrealm.demography.canopy_functions import calculate_canopy_q_m

    actual_q_m_values = calculate_canopy_q_m(
        m=fixture_canopy_shape["m"], n=fixture_canopy_shape["n"]
    )

    assert np.allclose(actual_q_m_values, fixture_canopy_shape["q_m"])


def test_calculate_canopy_z_max_proportion(fixture_canopy_shape):
    """Test calculate_canopy_z_max_proportion."""

    from pyrealm.demography.canopy_functions import calculate_canopy_z_max_proportion

    actual_p_zm = calculate_canopy_z_max_proportion(
        m=fixture_canopy_shape["m"], n=fixture_canopy_shape["n"]
    )

    assert np.allclose(actual_p_zm, fixture_canopy_shape["p_zm"])


@pytest.mark.parametrize(
    argnames="crown_areas, expected_r0",
    argvalues=(
        (np.array([20, 30]), np.array([0.86887756, 1.29007041])),
        (np.array([30, 40]), np.array([1.06415334, 1.489645])),
    ),
)
def test_calculate_r_0_values(fixture_canopy_shape, crown_areas, expected_r0):
    """Test happy path for calculating r_0."""

    from pyrealm.demography.canopy_functions import calculate_canopy_r0

    actual_r0_values = calculate_canopy_r0(
        q_m=fixture_canopy_shape["q_m"], crown_area=crown_areas
    )

    assert np.allclose(actual_r0_values, expected_r0)


def test_calculate_relative_canopy_radius_at_z(fixture_community):
    """Test crown radius height prediction.

    This test validates the expectation that the canopy shape model correctly
    predicts the crown area from the T Model equations at the predicted height of
    maximum crown radius.
    """

    from pyrealm.demography.canopy_functions import (
        calculate_relative_canopy_radius_at_z,
    )

    # Canopy shape model gives the maximum radius at a height z_max
    z_max = (
        fixture_community.cohort_data["stem_height"]
        * fixture_community.cohort_data["z_max_prop"]
    )

    # Get the relative radius at that height
    q_z_values = calculate_relative_canopy_radius_at_z(
        z=z_max,
        stem_height=fixture_community.cohort_data["stem_height"],
        m=fixture_community.cohort_data["m"],
        n=fixture_community.cohort_data["n"],
    )

    # Now test that the circular crown area from that radius is equivalent to the direct
    # prediction from the T model allometric equations.
    assert np.allclose(
        fixture_community.cohort_data["crown_area"],
        np.pi * (q_z_values * fixture_community.cohort_data["canopy_r0"]) ** 2,
    )


@pytest.mark.parametrize(
    argnames="heights,expected_Ap_z",
    argvalues=[
        pytest.param(
            np.array([15.19414157, 21.27411267, 23.70702725, 24.68056368]) + 0.01,
            np.repeat(0, 4),
            id="one_cm_above_stem_top",
        ),
        pytest.param(
            np.array([12.91932028, 18.08901635, 20.15768226, 20.98546374]) + 1.00,
            np.array([5.94793264, 19.6183899, 33.77430339, 47.31340371]),
            id="one_metre_above_z_max",
        ),
        pytest.param(
            np.array([12.91932028, 18.08901635, 20.15768226, 20.98546374]),
            np.array([8.03306419, 22.49502702, 37.60134866, 52.19394627]),
            id="at_z_max",
        ),
        pytest.param(
            np.array([12.91932028, 18.08901635, 20.15768226, 20.98546374]) - 1.00,
            np.array([8.03306419, 22.49502702, 37.60134866, 52.19394627]),
            id="one_metre_below_z_max",
        ),
    ],
)
def test_calculate_stem_projected_canopy_area_at_z(
    fixture_community, heights, expected_Ap_z
):
    """Test calculate_stem_projected_canopy_area_at_z.

    The test checks cases:
    * above stem H - all values should be zero
    * 1 metre above z_max - all values should be less than crown area
    * at z_max - all values should be equal to crown area
    * 1 metre below z_max - all values should be equal to crown area
    """

    from pyrealm.demography.canopy_functions import (
        calculate_stem_projected_canopy_area_at_z,
    )

    Ap_z_values = calculate_stem_projected_canopy_area_at_z(
        z=heights,
        stem_height=fixture_community.cohort_data["stem_height"],
        crown_area=fixture_community.cohort_data["crown_area"],
        m=fixture_community.cohort_data["m"],
        n=fixture_community.cohort_data["n"],
        q_m=fixture_community.cohort_data["q_m"],
        z_m=fixture_community.cohort_data["canopy_z_max"],
    )

    assert np.allclose(
        Ap_z_values,
        expected_Ap_z,
    )


def test_solve_community_projected_canopy_area(fixture_community):
    """Test solve_community_projected_canopy_area.

    The logic of this test is that given the cumulative sum of the crown areas in the
    fixture from tallest to shortest as the target, providing the z_max of each stem as
    the height _should_ always return zero, as this is exactly the height at which that
    cumulative area would close: crown 1 closes at z_max 1, crown 1 + 2 closes at z_max
    2 and so on.
    """

    from pyrealm.demography.canopy_functions import (
        solve_community_projected_canopy_area,
    )

    for (
        this_height,
        this_target,
    ) in zip(
        np.flip(fixture_community.cohort_data["canopy_z_max"]),
        np.cumsum(np.flip(fixture_community.cohort_data["crown_area"])),
    ):
        solved = solve_community_projected_canopy_area(
            z=this_height,
            stem_height=fixture_community.cohort_data["stem_height"],
            crown_area=fixture_community.cohort_data["crown_area"],
            n_individuals=fixture_community.cohort_data["n_individuals"],
            m=fixture_community.cohort_data["m"],
            n=fixture_community.cohort_data["n"],
            q_m=fixture_community.cohort_data["q_m"],
            z_m=fixture_community.cohort_data["canopy_z_max"],
            target_area=this_target,
        )

    assert solved == pytest.approx(0)


def test_calculate_projected_leaf_area_for_individuals():
    """Test happy path for calculating projected leaf area for individuals."""
    pass


def test_calculate_total_canopy_A_cp():
    """Test happy path for calculating total canopy A_cp across a community."""
    pass
