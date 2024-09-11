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
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": "a",
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2, 10, 3],
                "cohort_dbh_values": [0.2, 0.1, 0.2],
            },
            pytest.raises(ValidationError),
            {"cell_area": ["Not a valid number."]},
            id="wrong_type_in_simple_field",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2, 10.2, 3],
                "cohort_dbh_values": [0.2, 0.1, 0.2],
            },
            pytest.raises(ValidationError),
            {"cohort_n_individuals": {1: ["Not a valid integer."]}},
            id="float_in_n_individuals",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2, 10, 3],
                "cohort_dbh_values": [0.2, "a", 0.2],
            },
            pytest.raises(ValidationError),
            {"cohort_dbh_values": {1: ["Not a valid number."]}},
            id="float_in_n_individuals",
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

    # Note that value.messages is an extension provided by marshmallow.ValidationError
    assert excep.value.messages == excep_message


@pytest.mark.parametrize(
    argnames="file_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            """cell_id,cell_area,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
1,100,broadleaf,0.25,6
1,100,broadleaf,0.3,3
1,100,broadleaf,0.35,1
1,100,conifer,0.5,1
1,100,conifer,0.6,1
""",
            does_not_raise(),
            None,
            id="correct",
        ),
        pytest.param(
            """cell_id,cell_elevation,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
1,100,broadleaf,0.25,6
1,100,broadleaf,0.3,3
1,100,broadleaf,0.35,1
1,100,conifer,0.5,1
1,100,conifer,0.6,1
""",
            pytest.raises(ValueError),
            "Missing fields in community data: cell_area",
            id="missing_field",
        ),
        pytest.param(
            """cell_id,cell_area,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
1,100,broadleaf,0.25,6
1,100,broadleaf,0.3,3
1,100,broadleaf,0.35,1
11,100,conifer,0.5,1
1,100,conifer,0.6,1
""",
            pytest.raises(ValueError),
            "Multiple cell id values fields in community data, see load_communities",
            id="not_just_one_cell_id",
        ),
        pytest.param(
            """cell_id,cell_area,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
1,100,broadleaf,0.25,6
1,100,broadleaf,0.3,3
1,100,broadleaf,0.35,1
1,200,conifer,0.5,1
1,100,conifer,0.6,1
""",
            pytest.raises(ValueError),
            "Cell area varies in community data",
            id="not_just_one_cell_area",
        ),
    ],
)
def test_Community_from_csv(tmp_path, fixture_flora, file_data, outcome, excep_message):
    """Test that a community can be successfully imported from a csv."""

    from pyrealm.demography.community import Community

    temp_file = tmp_path / "data.csv"
    temp_file.write_text(file_data, encoding="utf-8")

    with outcome as excep:
        community = Community.from_csv(path=temp_file, flora=fixture_flora)

        if isinstance(outcome, does_not_raise):
            # TODO - test something here
            assert community
            return

    assert str(excep.value) == excep_message
