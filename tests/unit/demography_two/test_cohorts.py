"""Testing the cohorts module."""

from contextlib import nullcontext as does_not_raise
from importlib import resources

import pandas as pd
import pytest
from pydantic import ValidationError


@pytest.mark.parametrize(
    argnames="inputs,outcome,msg",
    argvalues=(
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [12]},
            does_not_raise(),
            None,
            id="valid",
        ),
        pytest.param(
            {
                "pft_name": ["name"],
                "dbh_value": [1.0],
                "n_individuals": [12],
                "community_id": [1],
            },
            does_not_raise(),
            None,
            id="valid_with_community",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0]},
            pytest.raises(ValidationError),
            "Field required",
            id="missing_field",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [12.2]},
            pytest.raises(ValidationError),
            "Input should be a valid integer",
            id="non_integer_n_individuals",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [-1.0], "n_individuals": [12]},
            pytest.raises(ValidationError),
            "Input should be greater than 0",
            id="negative_dbh",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [0]},
            pytest.raises(ValidationError),
            "Input should be greater than 0",
            id="no_individuals",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [12, 12]},
            pytest.raises(ValidationError),
            "Unequal field lengths",
            id="unequal_lengths",
        ),
    ),
)
def test_CohortData(inputs, outcome, msg):
    """Test the CohortData validation model."""
    from pyrealm.demography_two.cohorts import CohortData

    with outcome as err_handler:
        data = CohortData.model_validate(inputs)

        if "community_id" in inputs:
            assert data.community_id == [1]
        else:
            assert data.community_id is None

        return

    assert err_handler.match(msg)


@pytest.mark.parametrize(
    argnames="filename,outcome,expect_community",
    argvalues=[
        pytest.param("cohorts.csv", does_not_raise(), True, id="correct"),
        pytest.param(
            "cohorts_no_community_id.csv",
            does_not_raise(),
            False,
            id="correct_no_community",
        ),
    ],
)
def test_CohortData_from_csv(filename, outcome, expect_community):
    """Test CohortData loading and conversion to DataFrame.

    This also checks that the optional community_id field is handled.
    """
    from pyrealm.demography_two.cohorts import CohortData

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        cohort_data = CohortData.from_csv(datapath)

        df = cohort_data.to_dataframe()

        assert ("community_id" in df) == expect_community


@pytest.mark.parametrize(
    argnames="filename,outcome,expect_community",
    argvalues=[
        pytest.param("cohorts.csv", does_not_raise(), True, id="correct"),
        pytest.param(
            "cohorts_no_community_id.csv",
            does_not_raise(),
            False,
            id="correct_no_community",
        ),
    ],
)
def test_Cohorts_from_csv(filename, outcome, expect_community):
    """Test CohortData loading and conversion to DataFrame.

    This also checks that the optional community_id field is handled correctly and that
    the resulting cohorts data frame includes the merged trait data.
    """
    from pyrealm.demography_two.cohorts import Cohorts
    from pyrealm.demography_two.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    flora = Flora(name=["test1", "test2"])

    with outcome:
        cohort_data = Cohorts.from_csv(datapath, flora=flora)

        assert ("community_id" in cohort_data.cohorts) == expect_community
        assert "tau_f" in cohort_data.cohorts
        assert "cohort_id" in cohort_data.cohorts


def test_Cohorts_with_Flora_extensibility():
    """Test that extended Flora subclasses work with Cohorts."""
    from pyrealm.demography_two.cohorts import Cohorts
    from pyrealm.demography_two.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / "cohorts.csv"

    # A new subclass with additional variables
    class FloraExtended(Flora):
        my_new_field: list[int] = [42]  # type: ignore[annotation-unchecked]  # noqa: RUF012

    flora = FloraExtended(name=["test1", "test2"])

    cohort_data = Cohorts.from_csv(datapath, flora=flora)

    assert "tau_f" in cohort_data.cohorts
    assert "my_new_field" in cohort_data.cohorts

    assert cohort_data.cohorts["my_new_field"].equals(
        pd.Series([42] * cohort_data.cohorts.shape[0])
    )
