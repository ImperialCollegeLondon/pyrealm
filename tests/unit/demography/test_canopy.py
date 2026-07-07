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

    The simple four layer test uses three identical cohorts to give some easily defined
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
    df = instance.to_dataframe()

    assert df.shape == (
        np.prod(cohort_args["projected_leaf_area"].shape),
        len(instance._array_attrs),
    )

    assert set(instance._array_attrs) == set(df.columns)

    # Test the inherited to_pandas method of the community canopy data
    df = instance.community_data.to_dataframe()

    assert df.shape == (
        len(instance.community_data.transmission_profile),
        len(instance.community_data._array_attrs),
    )

    assert set(instance.community_data._array_attrs) == set(df.columns)


def test_Canopy__init__():
    """Test initialisation.

    Test that when a new canopy object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.canopy import Canopy
    from pyrealm.demography.cohorts import cohort_id_generator, create_cohorts
    from pyrealm.demography.flora import Flora
    from pyrealm.demography.tmodel import StemAllometry

    flora = Flora(name=["broadleaf", "conifer"], h_max=[30, 20])

    cid_gen = cohort_id_generator()
    cohorts = create_cohorts(
        pft_name=np.array(["broadleaf", "conifer"]),
        n_individuals=np.array([6, 1]),
        dbh_value=np.array([0.2, 0.5]),
        flora=flora,
        cid_generator=cid_gen,
    )

    allometry = StemAllometry(cohorts=cohorts)

    canopy_gap_fraction = 0.05
    cell_area = 20
    canopy = Canopy(
        cohorts=cohorts,
        allometry=allometry,
        canopy_area=cell_area,
        canopy_gap_fraction=canopy_gap_fraction,
        fit_ppa=True,
    )

    # Simply check that the shape of the stem leaf area matrix is the right shape
    n_layers_from_crown_area = int(
        np.ceil(
            (
                (allometry.crown_area * cohorts.n_individuals).sum()
                * (1 + canopy_gap_fraction)
            )
            / cell_area
        )
    )
    assert canopy.cohort_data.stem_leaf_area.shape == (
        n_layers_from_crown_area,
        canopy.n_cohorts,
    )


def test_solve_canopy_area_filling_height(fixture_cohorts_and_allometry):
    """Test solve_community_projected_canopy_area.

    The logic of this test is that given the cumulative sum of the crown areas in the
    fixture from tallest to shortest as the target, providing the z_max of each stem as
    the height _should_ always return zero, as this is exactly the height at which that
    cumulative area would close: crown 1 closes at z_max 1, crown 1 + 2 closes at z_max
    2 and so on.
    """

    from pyrealm.demography.canopy import solve_canopy_area_filling_height

    cohorts, allometry = fixture_cohorts_and_allometry

    for (
        this_height,
        this_target,
    ) in zip(
        np.flip(allometry.crown_z_max.flatten()),
        np.cumsum(np.flip(allometry.crown_area)),
    ):
        solved = solve_canopy_area_filling_height(
            z=this_height,
            stem_height=allometry.stem_height,
            crown_area=allometry.crown_area,
            n_individuals=cohorts.n_individuals,
            m=cohorts.m,
            n=cohorts.n,
            q_m=cohorts.q_m,
            z_max=allometry.crown_z_max,
            target_area=this_target,
        )

    assert solved == pytest.approx(0)


def test_fit_perfect_plasticity_approximation():
    """Test the PPA solver.

    The logic of this test is to construct a community of 3 singleton cohorts with DBH
    at 10, 20, 30 metres, but where the PFT traits are back-calculated so that each stem
    has the same 60 metre crown area. The PPA solver should then find that the layer
    closure occurs at the stems heights where the crown is widest (crown_z_max).

    This does rely on the canopy shape being flat topped enough that the crowns do not
    overlap vertically.
    """

    from pyrealm.demography.canopy import fit_perfect_plasticity_approximation
    from pyrealm.demography.cohorts import cohort_id_generator, create_cohorts
    from pyrealm.demography.flora import Flora
    from pyrealm.demography.tmodel import StemAllometry, calculate_dbh_from_height

    # Calculate DBH from target stem heights
    stem_height = np.array([30, 20, 10])
    h_max = np.array([35, 35, 35])
    a_hd = np.array([116, 116, 116])
    dbh = calculate_dbh_from_height(h_max=h_max, a_hd=a_hd, stem_height=stem_height)

    # Set the desired cell area and then back-calculate the ca_ratio trait to give each
    # stem that crown area
    area = 60
    ca_ratio = (4 * a_hd * area) / (np.pi * stem_height * dbh)

    # Set up the inputs.
    flora = Flora(
        name=["a", "b", "c"],
        h_max=list(h_max),
        a_hd=list(a_hd),
        ca_ratio=list(ca_ratio),
    )

    cid_gen = cohort_id_generator()
    cohorts = create_cohorts(
        pft_name=np.array(["a", "b", "c"]),
        dbh_value=dbh,
        n_individuals=np.ones(3).astype(int),
        flora=flora,
        cid_generator=cid_gen,
    )

    allometry = StemAllometry(cohorts)

    # Fit the model
    tolerance = 1e-8
    heights = fit_perfect_plasticity_approximation(
        cohorts=cohorts,
        allometry=allometry,
        area=area,
        canopy_gap_fraction=0,
        max_stem_height=30,
        solver_tolerance=tolerance,
    )

    # Add the final zero to represent remaining gap to ground.
    assert np.allclose(
        heights, np.concatenate([allometry.crown_z_max, [0]]), atol=tolerance
    )
