"""Test the Flora object."""

from contextlib import nullcontext as does_not_raise
from importlib import resources

import pandas as pd
import pytest
from pandas.errors import ParserError
from pydantic import ValidationError


@pytest.fixture
def flora_data(mode: str, length: int, unequal: bool):
    """Fixture providing inputs for Flora."""

    if mode == "empty":
        return {}

    args: dict[str, list] = dict(
        name=["defaults"],
        a_hd=[116.0],
        ca_ratio=[390.43],
        h_max=[15.33],
        lai=[1.8],
        par_ext=[0.5],
        resp_f=[0.1],
        resp_r=[0.913],
        resp_s=[0.044],
        resp_rt=[0],
        tau_rt=[0],
        rho_s=[200.0],
        sla=[14.0],
        tau_f=[4.0],
        tau_r=[1.04],
        yld=[0.17],
        zeta=[0.17],
        f_g=[0.02],
        m=[2],
        n=[5],
        p_foliage_for_reproductive_tissue=[0],
        gpp_topslice=[0],
    )

    if mode == "partial":
        args = {k: v for k, v in args.items() if k not in ["tau_f", "tau_"]}

    if length > 1:
        args = {k: v * length for k, v in args.items()}

    if unequal:
        args["f_g"] = args["f_g"] * 2

    return args


@pytest.mark.parametrize(
    argnames="mode,length,unequal,strict,outcome,msg",
    argvalues=(
        pytest.param("empty", 1, False, False, does_not_raise(), None, id="empty_lax"),
        pytest.param(
            "partial", 1, False, False, does_not_raise(), None, id="partial_lax"
        ),
        pytest.param("full", 1, False, False, does_not_raise(), None, id="full_lax"),
        pytest.param(
            "empty",
            1,
            False,
            True,
            pytest.raises(ValidationError),
            "Missing traits in strict mode",
            id="empty_strict",
        ),
        pytest.param(
            "partial",
            1,
            False,
            True,
            pytest.raises(ValidationError),
            "Missing traits in strict mode",
            id="partial_strict",
        ),
        pytest.param("full", 1, False, True, does_not_raise(), None, id="full_strict"),
        pytest.param(
            "full", 2, False, True, does_not_raise(), None, id="full_strict_2"
        ),
        pytest.param(
            "partial", 2, False, False, does_not_raise(), None, id="partial_lax_2"
        ),
        pytest.param(
            "partial",
            1,
            True,
            False,
            pytest.raises(ValidationError),
            "Unequal field lengths",
            id="partial_unequal",
        ),
        pytest.param(
            "full",
            1,
            True,
            False,
            pytest.raises(ValidationError),
            "Unequal field lengths",
            id="full_unequal",
        ),
    ),
)
def test_Flora(flora_data, mode, length, unequal, strict, outcome, msg):
    """Test the validation for Flora.

    Checks the strict and lax modes work as expected with empty, partial and full inputs
    and checks the unequal length validation works.
    """

    from pyrealm.demography_two.flora import Flora

    # Would like to use spy here to validate that field validation failures exit before
    # running custom model validation, but the line below always seems to return
    # call_count = 0
    # spy = mocker.spy(Flora, "strict_validation")

    with outcome as err_handler:
        v = Flora.model_validate(flora_data, context={"strict": strict})

        # Missing fields from partial and empty modes are present and the right length
        assert v.tau_f == [4.0] * length
        assert v.tau_r == [1.04] * length

        # Computed fields are present
        assert hasattr(v, "q_m")
        assert hasattr(v, "z_max_prop")

        return

    # Check errors raise the expected message
    assert err_handler.match(msg)


def test_Flora_unique_names():
    """Check the unique name constraint fires."""
    from pyrealm.demography_two.flora import Flora

    with pytest.raises(ValidationError):
        _ = Flora(name=["duplicated", "duplicated"])


@pytest.mark.parametrize(
    argnames="filename,strict,outcome",
    argvalues=[
        pytest.param("pfts.csv", False, does_not_raise(), id="correct"),
        pytest.param("pfts.json", False, pytest.raises(ParserError), id="format_wrong"),
        pytest.param(
            "no.pfts", False, pytest.raises(FileNotFoundError), id="file_missing"
        ),
        pytest.param("pfts_partial.csv", False, does_not_raise(), id="partial_lax"),
        pytest.param(
            "pfts_partial.csv",
            True,
            pytest.raises(ValidationError),
            id="partial_strict",
        ),
        pytest.param(
            "pfts_invalid.csv", False, pytest.raises(ValidationError), id="invalid"
        ),
    ],
)
def test_Flora_from_csv(filename, strict, outcome):
    """Test CSV loading."""
    from pyrealm.demography_two.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = Flora.from_csv(datapath, strict=strict)
        assert flora.name == ["test1", "test2"]


def test_Flora_to_dataframe():
    """Test conversion to dataframe."""
    from pyrealm.demography_two.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / "pfts.csv"
    flora = Flora.from_csv(datapath)

    df = flora.to_dataframe()

    assert df["name"].equals(pd.Series(["test1", "test2"]))
    assert df["a_hd"].equals(pd.Series([116.0, 116.0]))


@pytest.mark.parametrize(
    argnames="mode,length,unequal",
    argvalues=(pytest.param("full", 2, False, id="complete_data"),),
)
def test_Flora_extensibility(flora_data, mode, length, unequal):
    """Test the extensibility of the Flora model."""
    from pyrealm.demography_two.flora import Flora

    # A new subclass with additional variables
    class FloraExtended(Flora):
        my_new_field: list[int] = [42]  # type: ignore[annotation-unchecked]  # noqa: RUF012

    # Strict mode still works
    with pytest.raises(ValidationError):
        flora = FloraExtended.model_validate(flora_data, context={"strict": True})

    # Create an instance to check defaults are filled in when not strict
    flora = FloraExtended.model_validate(flora_data)

    assert hasattr(flora, "my_new_field")
    assert getattr(flora, "my_new_field") == [42, 42]

    # Check it works when data provided
    flora_data["my_new_field"] = [1, 1]

    flora = FloraExtended.model_validate(flora_data)

    assert hasattr(flora, "my_new_field")
    assert getattr(flora, "my_new_field") == [1, 1]

    # And that the field is present in the dataframe version.
    flora_df = flora.to_dataframe()

    assert "my_new_field" in flora_df
    assert flora_df["my_new_field"].equals(pd.Series([1, 1]))
