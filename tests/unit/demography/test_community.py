"""test the community object in community.py initialises as expected."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from marshmallow.exceptions import ValidationError


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
            # TODO - test something here
            assert community


@pytest.mark.parametrize(
    argnames="file_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10, 3]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            does_not_raise(),
            None,
            id="correct_numpy_arrays",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2, 10, 3],
                "cohort_dbh_values": [0.2, 0.1, 0.2],
            },
            does_not_raise(),
            None,
            id="correct_list",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2],
                "cohort_dbh_values": [0.2, 0.1, 0.2],
            },
            pytest.raises(ValidationError),
            {"_schema": ["Cohort arrays of unequal length."]},
            id="unequal_cohort_arrays",
        ),
        pytest.param(
            {
                "cell_area": 100,
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2, 10, 3],
                "cohort_dbh_values": [0.2, 0.1, 0.2],
            },
            pytest.raises(ValidationError),
            {"cell_id": ["Missing data for required field."]},
            id="missing_field",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cell_elevation": 100,
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2, 10, 3],
                "cohort_dbh_values": [0.2, 0.1, 0.2],
            },
            pytest.raises(ValidationError),
            {"cell_elevation": ["Unknown field."]},
            id="extra_field",
        ),
    ],
)
def test_Community__from_file_data(fixture_flora, file_data, outcome, excep_message):
    """Test happy path for initialisation.

    Test that when a new community object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.community import Community

    with outcome as excep:
        community = Community._from_file_data(flora=fixture_flora, file_data=file_data)

        if isinstance(outcome, does_not_raise):
            # TODO - test something here
            assert community
            return

    assert excep.value.messages == excep_message


def test_import_from_csv():
    """Test that a community can be successfully imported from a csv."""
    pass
