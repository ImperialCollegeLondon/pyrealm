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
    argnames="args,outcome,excep_message",
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
            id="correct",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": ["broadleaf", "broadleaf", "conifer"],
                "cohort_n_individuals": [2, 10, 3],
                "cohort_dbh_values": [0.2, 0.1, 0.2],
            },
            pytest.raises(ValueError),
            "Cohort data not passed as numpy arrays.",
            id="lists_not_arrays",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(ValueError),
            "Cohort arrays are of unequal length",
            id="unequal_cohort_arrays",
        ),
        pytest.param(
            {
                "cell_area": 100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10, 3]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(TypeError),
            "Community.__init__() missing 1 required positional argument: 'cell_id'",
            id="missing_arg",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cell_elevation": 100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10, 3]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(TypeError),
            "Community.__init__() got an unexpected keyword argument 'cell_elevation'",
            id="extra_arg",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": "100",
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(ValueError),
            "Community cell area must be a positive number.",
            id="cell_area_as_string",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": -100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10, 3]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(ValueError),
            "Community cell area must be a positive number.",
            id="cell_area_negative",
        ),
        pytest.param(
            {
                "cell_id": "1",
                "cell_area": 100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10, 3]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(ValueError),
            "Community cell id must be a integer >= 0.",
            id="cell_id_as_string",
        ),
        pytest.param(
            {
                "cell_id": -1,
                "cell_area": 100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "conifer"]),
                "cohort_n_individuals": np.array([2, 10, 3]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(ValueError),
            "Community cell id must be a integer >= 0.",
            id="cell_id_negative",
        ),
        pytest.param(
            {
                "cell_id": 1,
                "cell_area": 100,
                "cohort_pft_names": np.array(["broadleaf", "broadleaf", "juniper"]),
                "cohort_n_individuals": np.array([2, 10, 3]),
                "cohort_dbh_values": np.array([0.2, 0.1, 0.2]),
            },
            pytest.raises(ValueError),
            "Plant functional types unknown in flora: juniper",
            id="unknown_pft",
        ),
    ],
)
def test_Community__init__(fixture_flora, args, outcome, excep_message):
    """Test Community initialisation.

    Test that when a new community object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.community import Community

    with outcome as excep:
        community = Community(**args, flora=fixture_flora)

        if isinstance(outcome, does_not_raise):
            # TODO - test something here
            assert community
            return

    # Check exception message
    assert str(excep.value) == excep_message


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
            pytest.raises(ValidationError),
            "{'cell_area': ['Missing data for required field.'],"
            " 'cell_elevation': ['Unknown field.']}",
            id="mislabelled_field",
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

    # Test exception explanation
    assert str(excep.value) == excep_message


@pytest.mark.parametrize(
    argnames="file_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            """{"cell_id":1,"cell_area":100,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6},
            {"pft_name":"broadleaf","dbh_value":0.25,"n_individuals":6},
            {"pft_name":"broadleaf","dbh_value":0.3,"n_individuals":3},
            {"pft_name":"broadleaf","dbh_value":0.35,"n_individuals":1},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1},
            {"pft_name":"conifer","dbh_value":0.6,"n_individuals":1}]}""",
            does_not_raise(),
            None,
            id="correct",
        ),
    ],
)
def test_Community_from_json(
    tmp_path, fixture_flora, file_data, outcome, excep_message
):
    """Test that a community can be successfully imported from JSON."""

    from pyrealm.demography.community import Community

    temp_file = tmp_path / "data.json"
    temp_file.write_text(file_data, encoding="utf-8")

    with outcome as excep:
        community = Community.from_json(path=temp_file, flora=fixture_flora)

        if isinstance(outcome, does_not_raise):
            # TODO - test something here
            assert community
            return

    assert str(excep.value) == excep_message


@pytest.mark.parametrize(
    argnames="file_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            """cell_area = 100
cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.25
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.3
n_individuals = 3
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.35
n_individuals = 1
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"

[[cohorts]]
dbh_value = 0.6
n_individuals = 1
pft_name = "conifer"
""",
            does_not_raise(),
            None,
            id="correct",
        ),
    ],
)
def test_Community_from_toml(
    tmp_path, fixture_flora, file_data, outcome, excep_message
):
    """Test that a community can be successfully imported from JSON."""

    from pyrealm.demography.community import Community

    temp_file = tmp_path / "data.toml"
    temp_file.write_text(file_data, encoding="utf-8")

    with outcome as excep:
        community = Community.from_toml(path=temp_file, flora=fixture_flora)

        if isinstance(outcome, does_not_raise):
            # TODO - test something here
            assert community
            return

    assert str(excep.value) == excep_message
