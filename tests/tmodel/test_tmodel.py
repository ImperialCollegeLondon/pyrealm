"""Test TModel class.

Tests the init, grow_ttree and other methods of TModel.
"""

import csv
from contextlib import nullcontext as does_not_raise
from importlib import resources

import numpy as np
import pytest

# Fixtures: inputs and expected values from the original implementation in R


@pytest.fixture(scope="module")
def rvalues():
    """Fixture to load test inputs from file.

    This is a time series of growth using the default trait values in R, mapped to the
    internal property names used in TTree
    """
    from pyrealm.tmodel import TModelTraits

    datapath = resources.files("pyrealm_build_data.t_model") / "rtmodel_output.csv"

    with open(str(datapath)) as infile:
        rdr = csv.DictReader(infile, quoting=csv.QUOTE_NONNUMERIC)
        values = [v for v in rdr]

    name_map = (
        ("dD", "delta_d"),
        ("D", "diameter"),
        ("H", "height"),
        ("Ac", "crown_area"),
        ("Wf", "mass_fol"),
        ("Ws", "mass_stm"),
        ("Wss", "mass_swd"),
        ("GPP", "gpp_actual"),
        ("Rm1", "resp_swd"),
        ("Rm2", "resp_frt"),
        ("dWs", "delta_mass_stm"),
        ("dWfr", "delta_mass_frt"),
    )

    # copy values under R names to py names
    traits = TModelTraits()

    for row in values:
        for rnm, pynm in name_map:
            # Fix some scaling differences:
            if pynm == "delta_d":
                # The R tmodel implementation rescales reported delta_d as
                # a radial increase in millimetres, not diameter increase in metres
                row[pynm] = row[rnm] / 500
            elif pynm == "gpp_actual":
                # The R tmodel implementation slices off foliar respiration costs from
                # GPP before doing anything - the pyrealm.tmodel implementation keeps
                # this cost within the tree calculation
                row[pynm] = row[rnm] / (1 - traits.resp_f)
            else:
                row[pynm] = row[rnm]

    return values


@pytest.mark.parametrize(argnames="row", argvalues=np.arange(0, 100, 10))
def test_tmodel_init(rvalues, row):
    """Test the geometry calculation of __init__ using the exemplar R data."""

    from pyrealm.constants.tmodel_const import TModelTraits
    from pyrealm.tmodel import TTree

    ttree = TTree(traits=TModelTraits, diameters=rvalues[row]["diameter"])

    for geom_est in ("height", "crown_area", "mass_fol", "mass_stm", "mass_swd"):
        assert np.allclose(getattr(ttree, geom_est), rvalues[row][geom_est])


@pytest.mark.parametrize(argnames="row", argvalues=np.arange(0, 100, 10))
def test_tmodel_reset_diameters(rvalues, row):
    """Test the geometry calculation using reset_diameters using the exemplar R data."""

    from pyrealm.constants.tmodel_const import TModelTraits
    from pyrealm.tmodel import TTree

    ttree = TTree(diameters=0.001, traits=TModelTraits())
    ttree.reset_diameters(rvalues[row]["diameter"])

    for geom_est in ("height", "crown_area", "mass_fol", "mass_stm", "mass_swd"):
        assert np.allclose(getattr(ttree, geom_est), rvalues[row][geom_est])


def test_tmodel_init_array(rvalues):
    """Test geometry calculation using an __init__array using the exemplar R data."""

    from pyrealm.constants.tmodel_const import TModelTraits
    from pyrealm.tmodel import TTree

    diams = np.array([rw["diameter"] for rw in rvalues])
    ttree = TTree(diameters=diams, traits=TModelTraits)

    for geom_est in ("height", "crown_area", "mass_fol", "mass_stm", "mass_swd"):
        vals = [rw[geom_est] for rw in rvalues]
        assert np.allclose(getattr(ttree, geom_est), vals)


@pytest.mark.parametrize(
    argnames="sequence, raises",
    argvalues=[
        ("init_only", pytest.raises(RuntimeError)),
        ("init_grow", does_not_raise()),
        ("init_set", pytest.raises(RuntimeError)),
        ("init_grow_set", pytest.raises(RuntimeError)),
        ("init_grow_set_grow", does_not_raise()),
    ],
)
def test_tmodel_growth_access(sequence, raises):
    """Check masking of growth estimates.

    Tests that the accessors for growth properties are properly masked when not set
    or outdated.
    """
    from pyrealm.constants.tmodel_const import TModelTraits
    from pyrealm.tmodel import TTree

    ttree = TTree(diameters=0.1, traits=TModelTraits())

    if sequence == "init_only":
        pass
    elif sequence == "init_grow":
        ttree.calculate_growth(7)
    elif sequence == "init_set":
        ttree.reset_diameters(0.2)
    elif sequence == "init_grow_set":
        ttree.calculate_growth(7)
        ttree.reset_diameters(0.2)
    elif sequence == "init_grow_set_grow":
        ttree.calculate_growth(7)
        ttree.reset_diameters(0.2)
        ttree.calculate_growth(7)

    with raises:
        _ = ttree.delta_d


@pytest.mark.parametrize(argnames="row", argvalues=np.arange(0, 100, 10))
def test_tmodel_calculate_growth(rvalues, row):
    """Test calculate_growth with scalars.

    Runs a test of the tmodel.TTree against output from the R implementation.
    The values in the test come from simulating a 100 year run starting from
    a stem diameter of 0.1 and with an annual GPP value of 7. Each row in the
    file is the successive growth, but this test just runs some values from the
    sequence.
    """

    from pyrealm.constants.tmodel_const import TModelTraits
    from pyrealm.tmodel import TTree

    # create a tree with the initial diameter given in the row
    ttree = TTree(diameters=rvalues[row]["diameter"], traits=TModelTraits())
    ttree.calculate_growth(7)

    for growth_est in (
        "delta_d",
        "gpp_actual",
        "resp_swd",
        "resp_frt",
        "delta_mass_stm",
        "delta_mass_frt",
    ):
        assert np.allclose(getattr(ttree, growth_est), rvalues[row][growth_est])


@pytest.mark.profiling
def test_tmodel_calculate_growth_array(rvalues):
    """Test calculate_growth with an array."""

    from pyrealm.constants.tmodel_const import TModelTraits
    from pyrealm.tmodel import TTree

    # create a tree with the initial diameter given in the row
    diams = np.array([rw["diameter"] for rw in rvalues])
    ttree = TTree(diameters=diams, traits=TModelTraits)
    ttree.calculate_growth(7)

    for growth_est in (
        "delta_d",
        "gpp_actual",
        "resp_swd",
        "resp_frt",
        "delta_mass_stm",
        "delta_mass_frt",
    ):
        vals = [rw[growth_est] for rw in rvalues]
        assert np.allclose(getattr(ttree, growth_est), vals)


# @pytest.mark.parametrize(
#     "varname",
#     [
#         (0, ("dD", "delta_d")),
#         (1, ("D", "diameter")),
#         (2, ("H", "height")),
#         (3, ("Ac", "crown_area")),
#         (4, ("Wf", "mass_fol")),
#         (5, ("Ws", "mass_stm")),
#         (6, ("Wss", "mass_swd")),
#         (7, ("GPP", "gpp_actual")),
#         (8, ("Rm1", "resp_swd")),
#         (9, ("Rm2", "resp_frt")),
#         (10, ("dWs", "delta_mass_stm")),
#         (11, ("dWfr", "delta_mass_frt")),
#     ],
# )
# def test_grow_ttree(rvalues, pyvalues, varname):
#     """Runs a test of the tmodel.grow_ttree against the same R output.
#     In this case, the iteration process through time is tested. If the
#     previous test is successful then the values being fed forward in the
#     iteration _should_ be all fine, but this checks that the time iteration
#     is being run correctly in addition to the implementation of the model being
#     correct

#     Args:
#         values: Expected outputs from R implementation

#     """

#     idx, (rv, pyv) = varname
#     traits, pyvalues = pyvalues

#     # Get all the R values across timesteps into an array
#     r_var = np.array([rw[rv] for rw in rvalues])
#     # Get the matching py output
#     py_var = pyvalues[..., idx]

#     # Some implementation differences
#     if pyv == "delta_d":
#         # The R tmodel implementation rescales reported delta_d as
#         # a radial increase in millimetres.
#         py_var *= 500
#     elif pyv == "gpp_actual":
#         # The R tmodel implementation slices off foliar respiration costs from
#         # GPP before doing anything - the pyrealm.tmodel implementation keeps
#         # this cost within the tree calculation
#         py_var *= 1 - traits.resp_f

#     assert np.allclose(r_var, py_var)
