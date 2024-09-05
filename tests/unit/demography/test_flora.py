"""Test flora methods."""

import sys
from contextlib import nullcontext as does_not_raise
from importlib import resources
from json import JSONDecodeError

import pytest
from marshmallow.exceptions import ValidationError
from pandas.errors import ParserError

if sys.version_info[:2] >= (3, 11):
    from tomllib import TOMLDecodeError
else:
    from tomli import TOMLDecodeError

#
# Test PlantFunctionalType dataclass
#


@pytest.mark.parametrize(
    argnames="args,outcome",
    argvalues=[
        pytest.param({"name": "broadleaf"}, does_not_raise(), id="correct"),
        pytest.param({}, pytest.raises(TypeError), id="no_name"),
    ],
)
def test_PlantFunctionalType__init__(args, outcome):
    """Test the plant functional type initialisation."""

    from pyrealm.canopy_model.model.flora import (
        PlantFunctionalType,
        calculate_q_m,
        calculate_z_max_proportion,
    )

    with outcome:
        pft = PlantFunctionalType(**args)

        # Check name attribute and post_init attributes if instantiation succeeds.
        if isinstance(outcome, does_not_raise):
            assert pft.name == "broadleaf"
            # Expected values from defaults
            assert pft.q_m == calculate_q_m(m=2, n=5)
            assert pft.z_max_prop == calculate_z_max_proportion(m=2, n=5)


#
# Test Flora initialisation
#


@pytest.fixture()
def flora_inputs(request):
    """Fixture providing flora inputs for testing.

    This is using indirect parameterisation in the test largely to isolate the import of
    PlantFunctionalType within a method and to allow a single parameterised test to
    have a diverse set of inputs.
    """

    from pyrealm.canopy_model.model.flora import PlantFunctionalType

    broadleaf = PlantFunctionalType(name="broadleaf")
    conifer = PlantFunctionalType(name="conifer")

    match request.param:
        case "not_sequence":
            return "Notasequence"
        case "sequence_not_all_pfts":
            return [1, 2, 3]
        case "single_pft":
            return [broadleaf]
        case "multiple_pfts":
            return [broadleaf, conifer]
        case "duplicated_names":
            return [broadleaf, broadleaf]


@pytest.mark.parametrize(
    argnames="flora_inputs,outcome",
    argvalues=[
        pytest.param("not_sequence", pytest.raises(ValueError)),
        pytest.param("sequence_not_all_pfts", pytest.raises(ValueError)),
        pytest.param("single_pft", does_not_raise()),
        pytest.param("multiple_pfts", does_not_raise()),
        pytest.param("duplicated_names", pytest.raises(ValueError)),
    ],
    indirect=["flora_inputs"],
)
def test_Flora__init__(flora_inputs, outcome):
    """Test the plant functional type initialisation."""

    from pyrealm.canopy_model.model.flora import Flora

    with outcome:
        flora = Flora(pfts=flora_inputs)

        # Simple check that PFT instances are correctly keyed by name.
        if isinstance(outcome, does_not_raise):
            for k, v in flora.items():
                assert k == v.name


#
# Test Flora factory methods from JSON, TOML, CSV
#


@pytest.mark.parametrize(
    argnames="filename,outcome",
    argvalues=[
        pytest.param("pfts.json", does_not_raise(), id="correct"),
        pytest.param("pfts_partial.json", does_not_raise(), id="partial"),
        pytest.param("pfts.toml", pytest.raises(JSONDecodeError), id="format_wrong"),
        pytest.param("no.pfts", pytest.raises(FileNotFoundError), id="file_missing"),
        pytest.param("pfts_invalid.json", pytest.raises(ValidationError), id="invalid"),
    ],
)
def test_flora_from_json(filename, outcome):
    """Test JSON loading."""
    from pyrealm.canopy_model.model.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = Flora.from_json(datapath)

        if isinstance(outcome, does_not_raise):
            # Coarse check of what got loaded
            assert len(flora) == 2
            for nm in ["test1", "test2"]:
                assert nm in flora


@pytest.mark.parametrize(
    argnames="filename,outcome",
    argvalues=[
        pytest.param("pfts.toml", does_not_raise(), id="correct"),
        pytest.param("pfts_partial.toml", does_not_raise(), id="partial"),
        pytest.param("pfts.json", pytest.raises(TOMLDecodeError), id="format_wrong"),
        pytest.param("no.pfts", pytest.raises(FileNotFoundError), id="file_missing"),
        pytest.param("pfts_invalid.toml", pytest.raises(ValidationError), id="invalid"),
    ],
)
def test_flora_from_toml(filename, outcome):
    """Test TOML loading."""
    from pyrealm.canopy_model.model.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = Flora.from_toml(datapath)

        if isinstance(outcome, does_not_raise):
            # Coarse check of what got loaded
            assert len(flora) == 2
            for nm in ["test1", "test2"]:
                assert nm in flora


@pytest.mark.parametrize(
    argnames="filename,outcome",
    argvalues=[
        pytest.param("pfts.csv", does_not_raise(), id="correct"),
        pytest.param("pfts.json", pytest.raises(ParserError), id="format_wrong"),
        pytest.param("no.pfts", pytest.raises(FileNotFoundError), id="file_missing"),
        pytest.param("pfts_partial.csv", does_not_raise(), id="partial"),
        pytest.param("pfts_invalid.csv", pytest.raises(ValidationError), id="invalid"),
    ],
)
def test_flora_from_csv(filename, outcome):
    """Test CSV loading."""
    from pyrealm.canopy_model.model.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = Flora.from_csv(datapath)

        if isinstance(outcome, does_not_raise):
            # Coarse check of what got loaded
            assert len(flora) == 2
            for nm in ["test1", "test2"]:
                assert nm in flora


#
# Test PlantFunctionalType __post_init__ functions
#


@pytest.mark.parametrize(
    argnames="m,n,q_m",
    argvalues=[(2, 5, 2.9038988210485766), (3, 4, 2.3953681843215673)],
)
def test_calculate_q_m(m, n, q_m):
    """Test calculation of q_m."""

    from pyrealm.canopy_model.model.flora import calculate_q_m

    calculated_q_m = calculate_q_m(m, n)
    assert calculated_q_m == pytest.approx(q_m)


def test_calculate_q_m_values_raises_exception_for_invalid_input():
    """Test unhappy path for calculating q_m.

    Test that an exception is raised when invalid arguments are provided to the
    function.
    """

    pass


@pytest.mark.parametrize(
    argnames="m,n,z_max_ratio",
    argvalues=[(2, 5, 0.8502830004171938), (3, 4, 0.7226568811456053)],
)
def test_calculate_z_max_ratio(m, n, z_max_ratio):
    """Test calculation of z_max proportion."""

    from pyrealm.canopy_model.model.flora import calculate_z_max_proportion

    calculated_zmr = calculate_z_max_proportion(m, n)
    assert calculated_zmr == pytest.approx(z_max_ratio)
