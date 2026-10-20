"""Test the Flora object."""

from contextlib import nullcontext as does_not_raise
from importlib import resources

import numpy as np
import pandas as pd
import pytest
from pandas.errors import ParserError
from pydantic import ValidationError


@pytest.fixture
def flora_data(mode: str, length: int, unequal: bool):
    """Fixture providing inputs for Flora."""

    if mode == "empty":
        return {}

    args: dict[str, tuple] = dict(
        pft_name=("defaults",),
        a_hd=(116.0,),
        ca_ratio=(390.43,),
        h_max=(15.33,),
        lai=(1.8,),
        par_ext=(0.5,),
        resp_f=(0.1,),
        resp_r=(0.913,),
        resp_s=(0.044,),
        rho_s=(200.0,),
        sla=(14.0,),
        tau_f=(4.0,),
        tau_r=(1.04,),
        tau_b=(np.inf,),
        yld=(0.17,),
        zeta=(0.17,),
        f_g=(0.02,),
        m=(2,),
        n=(5,),
    )

    if mode == "partial":
        args = {k: v for k, v in args.items() if k not in ["tau_f", "tau_r"]}

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
class TestFloraValidation:
    """Test validation directly and through create_flora."""

    def test_FloraValidator(
        self, flora_data, mode, length, unequal, strict, outcome, msg
    ):
        """Test the FloraValidator.

        Checks the strict and lax modes work as expected with empty, partial and full
        inputs and checks the unequal length validation works.
        """

        from pyrealm.demography.flora import FloraValidator

        # Would like to use spy here to validate that field validation failures exit
        # before running custom model validation, but the line below always seems to
        # return call_count = 0
        # spy = mocker.spy(Flora, "strict_validation")

        with outcome as err_handler:
            v = FloraValidator.model_validate(flora_data, context={"strict": strict})

            # Missing fields from partial and empty modes are present and the right
            # length
            assert v.tau_f == (4.0,) * length
            assert v.tau_r == (1.04,) * length

            # Computed fields are present
            assert hasattr(v, "q_m")
            assert hasattr(v, "z_max_prop")

            return

        # Check errors raise the expected message
        assert err_handler.match(msg)

    def test_create_flora(
        self, flora_data, mode, length, unequal, strict, outcome, msg
    ):
        """Test the create_flora wrapper function using the same cases."""
        from pyrealm.demography.flora import create_flora

        with outcome as err_handler:
            v = create_flora(flora_data, strict=strict)

            # Missing fields from partial and empty modes are present and the right
            # length
            assert v["tau_f"].equals(pd.Series([4.0] * length))
            assert v["tau_r"].equals(pd.Series([1.04] * length))

            # Computed fields are present
            assert "q_m" in v.columns
            assert "z_max_prop" in v.columns

            return

        # Check errors raise the expected message
        assert err_handler.match(msg)


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
    """Test CSV loading, handling main error cases."""
    from pyrealm.demography.flora import load_flora_from_csv

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = load_flora_from_csv(datapath, strict=strict)
        assert flora["pft_name"].equals(pd.Series(["test1", "test2"]))
        assert flora["a_hd"].equals(pd.Series([116.0, 116.0]))


@pytest.mark.parametrize(
    argnames="mode,length,unequal",
    argvalues=(pytest.param("full", 2, False, id="complete_data"),),
)
def test_Flora_extensibility(flora_data, mode, length, unequal):
    """Test the extensibility of the Flora model."""
    from pyrealm.demography.flora import FloraValidator, create_flora

    # A new subclass with additional variables
    class FloraValidatorExtended(FloraValidator):
        my_new_field: tuple[int, ...] = (42,)

    # Strict mode still works both with direct usage and when passing in to create_flora
    with pytest.raises(ValidationError):
        flora = FloraValidatorExtended.model_validate(
            flora_data, context={"strict": True}
        )

    with pytest.raises(ValidationError):
        flora = create_flora(flora_data, strict=True, validator=FloraValidatorExtended)

    # Create an instance to check defaults are filled in when not strict
    flora = FloraValidatorExtended.model_validate(flora_data)

    assert hasattr(flora, "my_new_field")
    assert getattr(flora, "my_new_field") == (42, 42)

    flora = create_flora(flora_data, validator=FloraValidatorExtended)

    assert "my_new_field" in flora.columns
    assert flora["my_new_field"].equals(pd.Series([42, 42]))

    # Check it works when data provided
    flora_data["my_new_field"] = (1, 1)

    flora = FloraValidatorExtended.model_validate(flora_data)

    assert hasattr(flora, "my_new_field")
    assert getattr(flora, "my_new_field") == (1, 1)

    flora = create_flora(flora_data, validator=FloraValidatorExtended)

    assert "my_new_field" in flora.columns
    assert flora["my_new_field"].equals(pd.Series([1, 1]))
