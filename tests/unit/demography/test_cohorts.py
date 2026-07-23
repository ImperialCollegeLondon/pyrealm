"""Testing the cohorts module."""

from contextlib import nullcontext as does_not_raise
from importlib import resources

import numpy as np
import pandas as pd
import pytest


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
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [0]},
            does_not_raise(),
            None,
            id="valid with no individuals",
        ),
        pytest.param(
            {"pft_name": [], "dbh_value": [], "n_individuals": []},
            does_not_raise(),
            None,
            id="valid with empty lists",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [-1.0], "n_individuals": [12]},
            pytest.raises(ValueError),
            "DBH values must be strictly positive",
            id="negative_dbh",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [12.2]},
            pytest.raises(ValueError),
            "The number of individuals must be integers >= 0",
            id="non_integer_n_individuals",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [-50]},
            pytest.raises(ValueError),
            "The number of individuals must be integers >= 0",
            id="negative individuals",
        ),
        pytest.param(
            {"pft_name": ["name"], "dbh_value": [1.0], "n_individuals": [12, 12]},
            pytest.raises(ValueError),
            "All arrays must be of the same size",
            id="unequal_lengths",
        ),
        pytest.param(
            {
                "pft_name": [["name"], ["name"]],
                "dbh_value": [[1.0], [1.0]],
                "n_individuals": [[12], [12]],
            },
            pytest.raises(ValueError),
            "Inputs must be 1 dimensional arrays",
            id="not_1d",
        ),
    ),
)
def test_create_cohorts(inputs, outcome, msg):
    """Test the Cohorts validation."""
    from pyrealm.demography.cohorts import cohort_id_generator, create_cohorts
    from pyrealm.demography.flora import Flora

    flora = Flora(name=["name"])
    cid_gen = cohort_id_generator()
    inputs = {k: np.array(v) for k, v in inputs.items()}

    with outcome as err_handler:
        cohorts = create_cohorts(flora=flora, cid_generator=cid_gen, **inputs)

        if "community_id" in inputs:
            assert cohorts.community_id.equals(pd.Series([1]))
        else:
            assert "community_id" not in cohorts

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
def test_create_cohorts_from_csv(filename, outcome, expect_community):
    """Test create_cohorts_from_csv.

    This also checks that the optional community_id field is handled correctly and that
    the resulting cohorts data frame includes the merged trait data.
    """
    from pyrealm.demography.cohorts import (
        cohort_id_generator,
        create_cohorts_from_csv,
    )
    from pyrealm.demography.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename
    cid_gen = cohort_id_generator()
    flora = Flora(name=["test1", "test2"])

    with outcome:
        cohort_data = create_cohorts_from_csv(
            path=datapath, cid_generator=cid_gen, flora=flora
        )

        assert ("community_id" in cohort_data) == expect_community
        assert "tau_f" in cohort_data
        assert "cohort_id" in cohort_data


def test_Cohorts_with_Flora_extensibility():
    """Test that extended Flora subclasses work with Cohorts."""
    from pyrealm.demography.cohorts import (
        cohort_id_generator,
        create_cohorts_from_csv,
    )
    from pyrealm.demography.flora import Flora

    cid_gen = cohort_id_generator()
    datapath = resources.files("pyrealm_build_data.community") / "cohorts.csv"

    # A new subclass with additional variables
    class FloraExtended(Flora):
        my_new_field: list[int] = [42]  # type: ignore[annotation-unchecked]  # noqa: RUF012

    flora = FloraExtended(name=["test1", "test2"])

    cohort_data = create_cohorts_from_csv(
        path=datapath, cid_generator=cid_gen, flora=flora
    )

    assert "tau_f" in cohort_data
    assert "my_new_field" in cohort_data

    assert cohort_data["my_new_field"].equals(pd.Series([42] * cohort_data.shape[0]))
