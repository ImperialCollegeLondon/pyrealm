"""Test the functions in crown.py."""

from collections import namedtuple
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose

ZQZInput = namedtuple(
    "ZQZInput",
    ["z", "stem", "more_stem", "q_z", "outcome", "excep_msg", "output_shape"],
)
"""Simple named tuple to make inputs to array validation a bit clearer.

Contents:
* an input value for z
* a first stem property row array
* a list giving another stem property array
* a value for q_z array or None
* the validation outcome: a pytest.raises or does_not_raise context handler
* the start of the expected message on failure or None.
* the expected shape of successful output

The two stem property elements allow the number of properties to be controlled at
the test level (differing number of args for different functions) but also to
introduce inconsistent property lengths. To package these up for use with, for
example a total of 3 stem properties:

.. code:: python

    z, stem, more_stem, q_z, outcome, excep_msg, exp_shape = (
        fixture_z_qz_stem_properties
    )
    stem_args = {"stem_one": stem, "stem_two": more_stem, "stem_three": more_stem}
"""


@pytest.fixture
def fixture_z_qz_test_cases(which):
    """Fixture providing test combinations of trait, size (z) and q_z values.

    This fixture provides a menu of inputs that can be used by tests through indirect
    parameterisation to share a set of test cases of inputs for the z, stem properties
    q_z arguments and expected outcome and exception message. In each case, the returned
    value is a ZQZInput instance.

    The tests here are ordered in the execution order of validation, so failing
    inputs only provide non-None values for the elements required to trigger the fail.
    """

    match which:
        case "pass_stem_props":
            return ZQZInput(
                z=None,
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "fail_1D_z_not_congruent_with_stem":
            return ZQZInput(
                z=np.ones(5),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=pytest.raises(ValueError),
                excep_msg="The array shapes of the trait (4,) and "
                "size (5,) arguments are not congruent",
                output_shape=None,
            )
        case "fail_2D_z_not_congruent":
            return ZQZInput(
                z=np.ones((5, 2)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=pytest.raises(ValueError),
                excep_msg="The array shapes of the trait (4,) and "
                "size (5, 2) arguments are not congruent",
                output_shape=None,
            )
        case "pass_0D_z":
            return ZQZInput(
                z=np.array(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_scalar_z":
            return ZQZInput(
                z=np.ones(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_row_array_z":
            return ZQZInput(
                z=np.ones(4),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_2D_column_array_z":
            return ZQZInput(
                z=np.ones((5, 1)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(5, 4),
            )
        case "fail_0D_z_but_q_z_not_row":
            return ZQZInput(
                z=np.array(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(7),
                outcome=pytest.raises(ValueError),
                excep_msg="The broadcast shapes of the trait and size arguments "
                "(4,) are not congruent with the shape of the at_size "
                "arguments (7,)",
                output_shape=None,
            )
        case "fail_1D_scalar_z_but_q_z_not_row":
            return ZQZInput(
                z=np.ones(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones((6, 9)),
                outcome=pytest.raises(ValueError),
                excep_msg="The broadcast shapes of the trait and size arguments "
                "(4,) are not congruent with the shape of the at_size "
                "arguments (6, 9)",
                output_shape=None,
            )

        case "fail_2D_column_z_but_q_z_not_congruent":
            return ZQZInput(
                z=np.ones((5, 1)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones((6, 9)),
                outcome=pytest.raises(ValueError),
                excep_msg="The broadcast shapes of the trait and size arguments "
                "(5, 4) are not congruent with the shape of the at_size "
                "arguments (6, 9)",
                output_shape=None,
            )
        case "pass_0D_z_and_q_z_row":
            return ZQZInput(
                z=np.array(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(4),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_row_z_with_scalar_q_z":
            return ZQZInput(
                z=np.ones(4),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.array(1),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_scalar_z_and_q_z_row":
            return ZQZInput(
                z=np.ones(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(4),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_row_z_and_q_z_row":
            return ZQZInput(
                z=np.ones(4),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(4),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_2D_column_z_and_congruent_q_z":
            return ZQZInput(
                z=np.ones((5, 1)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones((5, 4)),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(5, 4),
            )


@pytest.mark.parametrize(
    argnames="which",
    argvalues=[
        "pass_0D_z",
        "pass_1D_scalar_z",
        "pass_1D_row_array_z",
        "pass_2D_column_array_z",
    ],
)
def test_calculate_relative_crown_radius_at_z_inputs(fixture_z_qz_test_cases, which):
    """Test calculate_relative_crown_radius_at_z input and output shapes .

    This test checks the function behaviour with different inputs.
    """

    from pyrealm.demography_two.crown import calculate_relative_crown_radius_at_z

    # Build inputs
    z, stem, more_stem, _, outcome, excep_msg, out_shape = fixture_z_qz_test_cases

    with outcome as excep:
        # Get the relative radius at that height
        q_z_values = calculate_relative_crown_radius_at_z(
            z=z, stem_height=z, m=stem, n=more_stem
        )

        if isinstance(outcome, does_not_raise):
            assert q_z_values.shape == out_shape

        return

    assert str(excep.value).startswith(excep_msg)


def test_calculate_relative_crown_radius_at_z_values(fixture_cohorts_and_allometry):
    """Test calculate_relative_crown_radius_at_z.

    This test validates the expectation that the canopy shape model correctly
    predicts the crown area from the T Model equations at the predicted height of
    maximum crown radius.
    """

    from pyrealm.demography_two.crown import (
        calculate_relative_crown_radius_at_z,
    )

    cohorts, allometry = fixture_cohorts_and_allometry

    # Get the relative radius at that heights of the crown z_max values
    q_z_values = calculate_relative_crown_radius_at_z(
        z=allometry.crown_z_max,
        stem_height=allometry.stem_height,
        m=cohorts.cohorts["m"].to_numpy(),
        n=cohorts.cohorts["n"].to_numpy(),
    )

    # Now test that the circular crown area from that radius is equivalent to
    # the direct prediction from the T model allometric equations.
    assert_allclose(
        allometry.crown_area,
        np.pi * (q_z_values * allometry.crown_r0) ** 2,
    )


@pytest.mark.parametrize(
    argnames="m,n",
    argvalues=[
        pytest.param(2, 5, id="default"),
        pytest.param(3, 1, id="conifer"),
        pytest.param(3, 1.5, id="tulip"),
        pytest.param(1, 3, id="inverse_conifer"),
        pytest.param(1.5, 3, id="inverse_tulip"),
    ],
)
def test_calculate_relative_crown_radius_at_z_values_clipping(m, n):
    """Test the clipping in calculate_relative_crown_radius_at_z.

    This test validates the expectation that the canopy shape model correctly
    predicts the crown area from the T Model equations at the predicted height of
    maximum crown radius.
    """

    from pyrealm.demography_two.crown import (
        calculate_relative_crown_radius_at_z,
    )
    from pyrealm.demography_two.flora import Flora

    # Default PFT settings
    flora = Flora(name=["test"], m=[m], n=[n])
    # Get the relative radius at that heights of the crown z_max values
    z_max_prop = flora.z_max_prop[0]
    q_z_values = calculate_relative_crown_radius_at_z(
        z=np.array(
            [
                11,  # Above the stem height
                10 * z_max_prop + 0.01,  # Just above the maximum width
                10 * z_max_prop,  # At the maximum width
                10 * z_max_prop + 0.01,  # Just below the maximum width
                -1,  # Below ground level
            ]
        )[:, None],
        stem_height=np.array([10]),
        m=np.array([flora.m]),
        n=np.array([flora.n]),
    )

    # Where z < 0 or z > H, q(z) should be zero
    assert_allclose(q_z_values[[0, 4]], np.zeros(2).reshape(2, 1))

    # Where z = H * z_max_prop, q(z) should be q_m
    assert q_z_values[2] == flora.q_m[0]

    # Either side of that, values should be >= 0 and < q_m
    assert np.all(np.less(q_z_values[[1, 3]], flora.q_m[0]))
    assert np.all(np.greater_equal(q_z_values[[1, 3]], 0))


@pytest.mark.parametrize(
    argnames="which",
    argvalues=[
        "pass_1D_row_z_with_scalar_q_z",
        "pass_0D_z_and_q_z_row",
        "pass_1D_scalar_z_and_q_z_row",
        "pass_1D_row_z_and_q_z_row",
        "pass_2D_column_z_and_congruent_q_z",
    ],
)
def test_calculate_stem_projected_crown_area_at_z_inputs(
    fixture_z_qz_test_cases, which
):
    """Tests the validation of inputs to calculate_stem_projected_crown_area_at_z."""
    from pyrealm.demography_two.crown import (
        calculate_stem_projected_crown_area_at_z,
    )

    # Build inputs
    z, stem, _more_stem, q_z, outcome, excep_msg, out_shape = fixture_z_qz_test_cases

    with outcome as excep:
        # Get the relative radius at that height
        Ap_z_values = calculate_stem_projected_crown_area_at_z(
            z=z, q_z=q_z, stem_height=z, crown_area=z, q_m=stem, z_max=z
        )

        if isinstance(outcome, does_not_raise):
            assert Ap_z_values.shape == out_shape

        return

    assert str(excep.value).startswith(excep_msg)


@pytest.mark.parametrize(
    argnames="heights,expected_Ap_z",
    argvalues=[
        pytest.param(
            np.array([15.19414157, 21.27411267, 23.70702725, 24.68056368]) + 0.01,
            np.array([0, 0, 0, 0]),
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
def test_calculate_stem_projected_crown_area_at_z_values(
    fixture_cohorts_and_allometry, heights, expected_Ap_z
):
    """Test calculate_stem_projected_canopy_area_at_z.

    The test checks cases:
    * above stem H - all values should be zero
    * 1 metre above z_max - all values should be less than crown area
    * at z_max - all values should be equal to crown area
    * 1 metre below z_max - all values should be equal to crown area
    """

    from pyrealm.demography_two.crown import (
        calculate_relative_crown_radius_at_z,
        calculate_stem_projected_crown_area_at_z,
    )

    cohorts, allometry = fixture_cohorts_and_allometry

    # Calculate the required q_z
    q_z = calculate_relative_crown_radius_at_z(
        z=heights,
        stem_height=allometry.stem_height,
        m=cohorts.cohorts["m"].to_numpy(),
        n=cohorts.cohorts["n"].to_numpy(),
    )

    # Calculate and test these values
    Ap_z_values = calculate_stem_projected_crown_area_at_z(
        z=heights,
        q_z=q_z,
        stem_height=allometry.stem_height,
        crown_area=allometry.crown_area,
        q_m=cohorts.cohorts["q_m"].to_numpy(),
        z_max=allometry.crown_z_max,
    )

    assert_allclose(
        Ap_z_values,
        expected_Ap_z,
    )


@pytest.mark.parametrize(
    argnames="which",
    argvalues=[
        "pass_1D_row_z_with_scalar_q_z",
        "pass_0D_z_and_q_z_row",
        "pass_1D_scalar_z_and_q_z_row",
        "pass_1D_row_z_and_q_z_row",
        "pass_2D_column_z_and_congruent_q_z",
    ],
)
def test_calculate_stem_projected_leaf_area_at_z_inputs(fixture_z_qz_test_cases, which):
    """Tests the validation of inputs to calculate_stem_projected_crown_area_at_z."""
    from pyrealm.demography_two.crown import (
        calculate_stem_projected_leaf_area_at_z,
    )

    # Build inputs
    z, stem, more_stem, q_z, outcome, excep_msg, out_shape = fixture_z_qz_test_cases

    with outcome as excep:
        # Get the relative radius at that height
        Ap_z_values = calculate_stem_projected_leaf_area_at_z(
            z=z,
            q_z=q_z,
            stem_height=z,
            crown_area=z,
            f_g=stem,
            q_m=more_stem,
            z_max=z,
        )

        if isinstance(outcome, does_not_raise):
            assert Ap_z_values.shape == out_shape

        return

    assert str(excep.value).startswith(excep_msg)


def test_calculate_stem_projected_leaf_area_at_z_values(fixture_cohorts_and_allometry):
    """Test calculate_stem_projected_leaf_area_at_z.

    This test uses hand calculated values to check predictions, but there are some more
    robust theoretical checks about the expectations and crown area.
    """

    from pyrealm.demography_two.crown import (
        calculate_relative_crown_radius_at_z,
        calculate_stem_projected_leaf_area_at_z,
    )

    cohorts, allometry = fixture_cohorts_and_allometry

    # Calculate the leaf areas at the locations of z_max for each stem from the lowest
    # to the highest
    z_max = allometry.crown_z_max[:, None]

    q_z = calculate_relative_crown_radius_at_z(
        z=z_max,
        stem_height=allometry.stem_height,
        m=cohorts.cohorts["m"].to_numpy(),
        n=cohorts.cohorts["n"].to_numpy(),
    )

    leaf_area_fg0 = calculate_stem_projected_leaf_area_at_z(
        z=z_max,
        q_z=q_z,
        stem_height=allometry.stem_height,
        crown_area=allometry.crown_area,
        f_g=cohorts.cohorts["f_g"].to_numpy(),
        q_m=cohorts.cohorts["q_m"].to_numpy(),
        z_max=allometry.crown_z_max,
    )

    # Pre-calculated values
    expected_leaf_area_fg0 = np.array(
        [
            [8.03306419, 22.49502702, 37.60134866, 52.19394627],
            [0.0, 22.49502702, 37.60134866, 52.19394627],
            [0.0, 9.67422125, 37.60134866, 52.19394627],
            [0.0, 1.04248076, 35.02960183, 52.19394627],
        ]
    )

    assert_allclose(leaf_area_fg0, expected_leaf_area_fg0)

    # More rigorous check - with f_g = 0, the projected leaf area of each stem in the
    # lowest layer must equal the crown area (all the crown is now accounted for).
    assert_allclose(leaf_area_fg0[0], allometry.crown_area)
    # Also the diagonal of the resulting matrix (4 heights for 4 cohorts) should _also_
    # match the crown areas as the leaf area is all accounted for exactly at z_max.
    assert_allclose(np.diag(leaf_area_fg0), allometry.crown_area)

    # Introduce some crown gap fraction and recalculate
    cohorts.cohorts["f_g"] += 0.02

    leaf_area_fg002 = calculate_stem_projected_leaf_area_at_z(
        z=z_max,
        q_z=q_z,
        stem_height=allometry.stem_height,
        crown_area=allometry.crown_area,
        f_g=cohorts.cohorts["f_g"].to_numpy(),
        q_m=cohorts.cohorts["q_m"].to_numpy(),
        z_max=allometry.crown_z_max,
    )

    expected_leaf_area_fg002 = np.array(
        [
            [7.8724029, 22.41196859, 37.5384868, 52.12953869],
            [0.0, 22.04512648, 37.03818313, 51.55306811],
            [0.0, 9.48073683, 36.84932168, 51.20070245],
            [0.0, 1.02163115, 34.32900979, 51.15006735],
        ]
    )

    assert_allclose(leaf_area_fg002, expected_leaf_area_fg002)

    # More rigorous checks:
    # - All leaf areas with f_g = 0.02 should be lower than with f_g = 0, accounting for
    #   zeros. TODO - this may change if the functions return np.nan above stem height.
    assert np.all(
        np.logical_or(np.less(leaf_area_fg002, leaf_area_fg0), leaf_area_fg0 == 0)
    )

    # - The diagonal should be exactly (1 - f_g) times the crown area: at the z_max for
    #   the stem all but the crown gap fraction should be accounted for
    assert_allclose(np.diag(leaf_area_fg002), allometry.crown_area * 0.98)


def test_CrownProfile(fixture_cohorts_and_allometry):
    """Test the CrownProfile class.

    This implements a subset of the tests in the more detailed function checks above to
    validate that this wrapper class works as intended.
    """

    from pyrealm.demography_two.crown import CrownProfile

    cohorts, allometry = fixture_cohorts_and_allometry

    # Estimate the profile at the heights of the maximum crown radii for each cohort

    crown_profile = CrownProfile(
        cohorts=cohorts,
        z=allometry.crown_z_max,
        allometry=allometry,
    )

    # Crown radius on diagonal predicts crown area accurately - needs to made 2D again.
    assert_allclose(
        np.diag(crown_profile.crown_radius) ** 2 * np.pi,
        allometry.crown_area,
    )

    # Same is true for projected crown area at z_max heights
    assert_allclose(
        np.diag(crown_profile.projected_crown_area),
        allometry.crown_area,
    )

    # And since f_g=0, so is projected leaf area
    assert_allclose(
        np.diag(crown_profile.projected_leaf_area),
        allometry.crown_area,
    )

    # Test the inherited to_pandas method
    df = crown_profile.to_dataframe()

    assert df.shape == (
        cohorts.n_cohorts * allometry.crown_z_max.size,
        len(crown_profile._array_attrs),
    )

    assert set(crown_profile._array_attrs) == set(df.columns)


@pytest.mark.parametrize(argnames="stem_offset", argvalues=[None, np.arange(4)])
@pytest.mark.parametrize(argnames="as_xy", argvalues=[True, False])
@pytest.mark.parametrize(argnames="two_sided", argvalues=[True, False])
def test_get_crown_xy(fixture_cohorts_and_allometry, as_xy, two_sided, stem_offset):
    """Test the get_crown_xy helper.

    This really just checks it runs at the moment.
    """

    from pyrealm.demography_two.crown import CrownProfile

    cohorts, allometry = fixture_cohorts_and_allometry

    # Estimate the profile at the heights of the maximum crown radii for each cohort

    crown_profile = CrownProfile(
        cohorts=cohorts,
        z=allometry.crown_z_max,
        allometry=allometry,
    )

    for attr in crown_profile._array_attrs:
        _ = crown_profile.to_xy(
            attr=attr, as_xy=as_xy, two_sided=two_sided, stem_offsets=stem_offset
        )
