"""test the community object in community.py initialises as expected."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.fixture
def fixture_flora():
    """Simple flora object for use in community tests."""

    from pyrealm.demography.flora import Flora, PlantFunctionalType

    return Flora(
        [
            PlantFunctionalType(name="broadleaf", h_max=30),
            PlantFunctionalType(name="conifer", h_max=20),
        ]
    )


@pytest.mark.parametrize(
    argnames="cell_id,cell_area,cohort_pfts,cohort_dbh,cohort_n,outcome",
    argvalues=[
        pytest.param(
            1,
            100,
            np.array(["broadleaf", "broadleaf", "conifer"]),
            np.array([0.2, 0.1, 0.2]),
            np.array([2, 10, 3]),
            does_not_raise(),
            id="correct",
        ),
        pytest.param(
            1,
            100,
            np.array(["broadleaf", "broadleaf", "juniper"]),
            np.array([0.2, 0.1, 0.2]),
            np.array([2, 10, 3]),
            pytest.raises(ValueError),
            id="unknown_pft",
        ),
        pytest.param(
            1,
            100,
            np.array(["broadleaf", "broadleaf", "juniper"]),
            np.array([0.2, 0.1, 0.2]),
            np.array([2, 10, 3, 4]),
            pytest.raises(ValueError),
            id="unequal_cohort_data_lengths",
        ),
    ],
)
def test_Community_initialisation(
    fixture_flora, cell_id, cell_area, cohort_pfts, cohort_dbh, cohort_n, outcome
):
    """Test happy path for initialisation.

    Test that when a new community object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.community import Community

    with outcome:
        community = Community(
            cell_id=cell_id,
            cell_area=cell_area,
            cohort_pft_names=cohort_pfts,
            cohort_dbh_values=cohort_dbh,
            cohort_n_individuals=cohort_n,
            flora=fixture_flora,
        )

        if isinstance(outcome, does_not_raise):
            assert community


def test_import_from_csv():
    """Test that a community can be successfully imported from a csv."""
    pass
