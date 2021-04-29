import os
import pytest
import numpy as np
from contextlib import contextmanager
import csv
from pyrealm import tmodel
from pyrealm import param_classes

# ------------------------------------------
# Null context manager to include exception testing in test paramaterisation
# ------------------------------------------

RPMODEL_C4_BUG = True

@contextmanager
def does_not_raise():
    yield


# ------------------------------------------
# Fixtures: inputs and expected values
# ------------------------------------------

@pytest.fixture(scope='module')
def rvalues():
    """Fixture to load test inputs from file.
    """

    test_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(test_dir, 'rtmodel_output.csv')) as infile:
        rdr = csv.DictReader(infile, quoting=csv.QUOTE_NONNUMERIC)
        values = [v for v in rdr]

    return values

@pytest.fixture(scope='module')
def pyvalues():
    """Run the same time series in python
    """

    # Do the same simulation as the R implementation - a single stem of
    # diameter 0.1 exposed to 100 years of GPP of 7.
    traits = param_classes.TModelTraits()
    py_vars = ('delta_d', 'diameter', 'height', 'crown_area', 'mass_fol',
               'mass_stm', 'mass_swd', 'gpp_actual', 'resp_swd', 'resp_frt',
               'delta_mass_stm', 'delta_mass_frt')
    values = tmodel.grow_ttree(np.array([[7] * 100]), np.array([0.1]), 1,
                               traits=traits, outvars=py_vars)

    return traits, values

# ------------------------------------------
# Tests
# ------------------------------------------


def test_ttree(rvalues):
    """ Runs a test of the tmodel.TTree against output from an R implementation.
    The values in the test come from simulating a 100 year run starting from
    a stem diameter of 0.1 and with an annual GPP value of 7. Each row in the
    file is the successive growth, but this test just runs each value in
    sequence.

    Args:
        values: Expected outputs from R implementation

    """

    for row in rvalues:

        # create a tree with the initial diameter given in the row
        tree = tmodel.TTree()
        tree.set_diameter(row['D'])
        tree.calculate_growth(7)

        # The R tmodel implementation rescales reported delta_d as
        # a radial increase in millimetres.
        assert np.allclose(tree.delta_d * 500, row['dD'])
        assert np.allclose(tree.diameter, row['D'])
        assert np.allclose(tree.height, row['H'])
        assert np.allclose(tree.crown_area, row['Ac'])
        assert np.allclose(tree.mass_fol, row['Wf'])
        assert np.allclose(tree.mass_stm, row['Ws'])
        assert np.allclose(tree.mass_swd, row['Wss'])
        # The R tmodel implementation slices off foliar respiration costs from
        # GPP before doing anything - the pyrealm.tmodel implementation keeps
        # this cost within the tree calculation
        assert np.allclose(tree.gpp_actual * (1 - tree.traits.resp_f), row['GPP'])
        assert np.allclose(tree.resp_swd, row['Rm1'])
        assert np.allclose(tree.resp_frt, row['Rm2'])
        assert np.allclose(tree.delta_mass_stm, row['dWs'])
        assert np.allclose(tree.delta_mass_frt, row['dWfr'])



@pytest.mark.parametrize(
    'varname',
    [(0, ('dD', 'delta_d')),
     (1, ('D', 'diameter')),
     (2, ('H', 'height')),
     (3, ('Ac', 'crown_area')),
     (4, ('Wf', 'mass_fol')),
     (5, ('Ws', 'mass_stm')),
     (6, ('Wss', 'mass_swd')),
     (7, ('GPP', 'gpp_actual')),
     (8, ('Rm1', 'resp_swd')),
     (9, ('Rm2', 'resp_frt')),
     (10, ('dWs', 'delta_mass_stm')),
     (11, ('dWfr', 'delta_mass_frt'))]
)
def test_grow_ttree(rvalues, pyvalues, varname):
    """ Runs a test of the tmodel.grow_ttree against the same R output.
    In this case, the iteration process through time is tested. If the
    previous test is successful then the values being fed forward in the
    iteration _should_ be all fine, but this checks that the time iteration
    is being run correctly in addition to the implementation of the model being
    correct

    Args:
        values: Expected outputs from R implementation

    """

    idx, (rv, pyv) = varname
    traits, pyvalues = pyvalues

    # Get all the R values across timesteps into an array
    r_var = np.array([rw[rv] for rw in rvalues])
    # Get the matching py output
    py_var = pyvalues[..., idx]

    # Some implementation differences
    if pyv == 'delta_d':
        # The R tmodel implementation rescales reported delta_d as
        # a radial increase in millimetres.
        py_var *= 500
    elif pyv == 'gpp_actual':
        # The R tmodel implementation slices off foliar respiration costs from
        # GPP before doing anything - the pyrealm.tmodel implementation keeps
        # this cost within the tree calculation
        py_var *= (1 - traits.resp_f)

    assert np.allclose(r_var, py_var)
